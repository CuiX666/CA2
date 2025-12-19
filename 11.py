import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from transformers import BertModel, BertTokenizer
import numpy as np
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from functools import partial
import time
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_outfit_mask(item_counts_per_outfit, max_items):
    batch_size = item_counts_per_outfit.size(0)
    mask = torch.arange(max_items, device=item_counts_per_outfit.device).expand(batch_size,
                                                                                max_items) < item_counts_per_outfit.unsqueeze(
        1)
    return mask


class ImageFeatureExtractor(nn.Module):
    def __init__(self, visual_feature_dim=512):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        self.output_dim = 512

    def forward(self, images):
        return self.resnet(images)


class TextFeatureExtractor(nn.Module):
    def __init__(self, text_feature_dim=768, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.bert = None
        self.tokenizer = None
        self.output_dim = text_feature_dim
        try:
            if os.path.isdir(bert_model_name) and bert_model_name != os.path.basename(bert_model_name):
                print(f"警告：本地目录 '{bert_model_name}' 存在。 "
                      f"HuggingFace Transformers 可能会尝试从中加载。 "
                      f"如果不是一个有效的模型目录，加载可能会失败或使用非预期的文件。")

            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert = BertModel.from_pretrained(bert_model_name)
            print(f"BERT模型 '{bert_model_name}' 加载成功。")

        except EnvironmentError as e:
            print(f"严重警告：无法从 Hugging Face Hub 或本地缓存加载 '{bert_model_name}'。 "
                  f"请确保网络连接或模型已在本地正确缓存。错误：{e}")
            print("后续的文本特征提取将产生零向量，除非问题解决且特征被重新计算。")

    def forward(self, texts_list, device):
        if self.tokenizer is None or self.bert is None:
            return torch.zeros(len(texts_list), self.output_dim, device=device)

        inputs = self.tokenizer(texts_list, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]


class OutfitAwareMappingParamsLearner(nn.Module):
    def __init__(self, combined_feature_dim, num_semantic_masks_K, semantic_feature_dim, mapped_feature_dim_per_mask):
        super().__init__()
        self.num_semantic_masks_K = num_semantic_masks_K
        self.semantic_feature_dim = semantic_feature_dim
        self.mapped_feature_dim_per_mask = mapped_feature_dim_per_mask

        self.context_encoder = nn.Sequential(
            nn.Linear(combined_feature_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.fc_params = nn.Linear(128, num_semantic_masks_K * (
                self.semantic_feature_dim * mapped_feature_dim_per_mask + mapped_feature_dim_per_mask))

    def forward(self, all_item_features_in_outfit, outfit_mask):
        masked_features = all_item_features_in_outfit * outfit_mask.unsqueeze(-1).float()
        sum_features = masked_features.sum(dim=1)
        item_counts = outfit_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-6)
        avg_outfit_context = sum_features / item_counts

        outfit_context_encoded = self.context_encoder(avg_outfit_context)
        raw_params = self.fc_params(outfit_context_encoded)

        Ws_batch, bs_batch = [], []
        current_idx = 0
        for _ in range(self.num_semantic_masks_K):
            W_k_size = self.semantic_feature_dim * self.mapped_feature_dim_per_mask
            b_k_size = self.mapped_feature_dim_per_mask

            W_k_flat = raw_params[:, current_idx: current_idx + W_k_size]
            current_idx += W_k_size
            b_k = raw_params[:, current_idx: current_idx + b_k_size]
            current_idx += b_k_size

            Ws_batch.append(W_k_flat.view(-1, self.semantic_feature_dim, self.mapped_feature_dim_per_mask))
            bs_batch.append(b_k)

        return Ws_batch, bs_batch


class OutfitAwareFineGrainedCompatibility(nn.Module):
    def __init__(self, combined_feature_dim, num_semantic_masks_K, mapped_feature_dim_per_mask, mlp_hidden_dim=128,
                 dropout_rate=0.3):
        super().__init__()
        self.combined_feature_dim = combined_feature_dim
        self.num_semantic_masks_K = num_semantic_masks_K
        self.mapped_feature_dim_per_mask = mapped_feature_dim_per_mask

        self.semantic_masks = nn.Parameter(torch.empty(num_semantic_masks_K, combined_feature_dim))
        nn.init.xavier_uniform_(self.semantic_masks)

        self.mapping_param_learner = OutfitAwareMappingParamsLearner(
            combined_feature_dim, num_semantic_masks_K, combined_feature_dim, mapped_feature_dim_per_mask
        )

        self.compatibility_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mapped_feature_dim_per_mask, mlp_hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(mlp_hidden_dim, 1)
            ) for _ in range(num_semantic_masks_K)])
        self.activation = nn.Sigmoid()

    def forward(self, item_features_cat, outfit_mask):
        if item_features_cat.isnan().any() or item_features_cat.isinf().any():
            return torch.zeros(item_features_cat.shape[0], 1, device=item_features_cat.device)

        batch_size, max_items, _ = item_features_cat.shape

        Ws_batch, bs_batch = self.mapping_param_learner(item_features_cat, outfit_mask)

        subspace_scores_s_k = []

        for k in range(self.num_semantic_masks_K):
            e_ik = item_features_cat * self.semantic_masks[k]

            current_W_k = Ws_batch[k]
            current_b_k = bs_batch[k]

            e_prime_ik = torch.einsum('bmd,bdo->bmo', e_ik, current_W_k) + current_b_k.unsqueeze(1)
            e_prime_ik = F.relu(e_prime_ik)

            masked_e_prime_ik = e_prime_ik * outfit_mask.unsqueeze(-1).float()
            sum_e_prime_ik = masked_e_prime_ik.sum(dim=1)

            item_counts_eff = outfit_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-6)
            avg_e_prime_ik_per_outfit = sum_e_prime_ik / item_counts_eff

            s_k_score = self.compatibility_mlps[k](avg_e_prime_ik_per_outfit)
            subspace_scores_s_k.append(s_k_score)

        all_s_k_scores = torch.cat(subspace_scores_s_k, dim=1)
        p_j = all_s_k_scores.mean(dim=1, keepdim=True)

        if p_j.isnan().any():
            p_j = torch.nan_to_num(p_j, nan=0.0)
        return self.activation(p_j)


class MultiViewCorrelationModule(nn.Module):
    def __init__(self, combined_feature_dim, max_items, dcca_output_dim=128, k_principal_components=10, r1=1e-4,
                 dropout_rate=0.3):
        super().__init__()
        self.combined_feature_dim = combined_feature_dim
        self.max_items = max_items
        self.dcca_output_dim = dcca_output_dim
        self.k_principal_components = k_principal_components
        self.r1 = r1

        input_flat_dim = max_items * combined_feature_dim

        self.f1_mlp = nn.Sequential(
            nn.LayerNorm(input_flat_dim),
            nn.Linear(input_flat_dim, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, dcca_output_dim)
        )
        self.f2_mlp = nn.Sequential(
            nn.LayerNorm(input_flat_dim),
            nn.Linear(input_flat_dim, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, dcca_output_dim)
        )
        self.activation = nn.Sigmoid()

    def forward(self, item_features_cat, outfit_mask):
        if item_features_cat.isnan().any() or item_features_cat.isinf().any():
            return (torch.zeros(item_features_cat.shape[0], 1, device=item_features_cat.device),
                    torch.eye(self.dcca_output_dim, device=item_features_cat.device))

        batch_size, _, _ = item_features_cat.shape

        masked_item_features = item_features_cat * outfit_mask.unsqueeze(-1).float()
        H_flat = masked_item_features.view(batch_size, -1)

        item_counts = outfit_mask.sum(dim=1)
        G_features_list = []
        for i in range(batch_size):
            count = int(item_counts[i].item())
            if count == 0:
                G_features_list.append(torch.zeros_like(masked_item_features[i]))
                continue
            valid_items = masked_item_features[i, :count, :]
            reversed_valid_items = torch.flip(valid_items, dims=[0])

            padded_reversed_items = torch.zeros_like(masked_item_features[i])
            padded_reversed_items[:count, :] = reversed_valid_items
            G_features_list.append(padded_reversed_items)

        if not G_features_list:
            return (torch.zeros(batch_size, 1, device=item_features_cat.device),
                    torch.eye(self.dcca_output_dim, device=item_features_cat.device))

        G_features = torch.stack(G_features_list, dim=0)
        G_flat = G_features.view(batch_size, -1)

        F_H = self.f1_mlp(H_flat)
        F_G = self.f2_mlp(G_flat)

        if F_H.isnan().any() or F_G.isnan().any():
            return (torch.zeros(batch_size, 1, device=F_H.device),
                    torch.eye(self.dcca_output_dim, device=F_H.device))

        F_H_mean = F_H - F_H.mean(dim=0, keepdim=True)
        F_G_mean = F_G - F_G.mean(dim=0, keepdim=True)

        N = batch_size

        if N <= self.dcca_output_dim or N < 2:
            return (torch.zeros(batch_size, 1, device=F_H.device),
                    torch.eye(self.dcca_output_dim, device=F_H.device))

        try:
            Sigma_HH = (1.0 / (N - 1)) * (F_H_mean.T @ F_H_mean) + self.r1 * torch.eye(self.dcca_output_dim,
                                                                                       device=F_H.device)
            Sigma_GG = (1.0 / (N - 1)) * (F_G_mean.T @ F_G_mean) + self.r1 * torch.eye(self.dcca_output_dim,
                                                                                       device=F_H.device)
            Sigma_HG = (1.0 / (N - 1)) * (F_H_mean.T @ F_G_mean)

            if Sigma_HH.isnan().any() or Sigma_GG.isnan().any() or Sigma_HG.isnan().any():
                return (torch.zeros(batch_size, 1, device=F_H.device),
                        torch.eye(self.dcca_output_dim, device=F_H.device))

            U_H, S_H_diag, V_H_T = torch.linalg.svd(Sigma_HH)
            Sigma_HH_inv_sqrt = U_H @ torch.diag(S_H_diag.clamp(min=1e-8).pow(-0.5)) @ V_H_T

            U_G, S_G_diag, V_G_T = torch.linalg.svd(Sigma_GG)
            Sigma_GG_inv_sqrt = U_G @ torch.diag(S_G_diag.clamp(min=1e-8).pow(-0.5)) @ V_G_T

            T_matrix = Sigma_HH_inv_sqrt @ Sigma_HG @ Sigma_GG_inv_sqrt

            if T_matrix.isnan().any():
                return (torch.zeros(batch_size, 1, device=F_H.device),
                        torch.eye(self.dcca_output_dim, device=F_H.device))

            singular_values_T = torch.linalg.svdvals(T_matrix)

        except RuntimeError as e:
            return (torch.zeros(batch_size, 1, device=F_H.device),
                    torch.eye(self.dcca_output_dim, device=F_H.device))

        actual_k_components = min(self.k_principal_components, singular_values_T.numel(), self.dcca_output_dim)

        if actual_k_components > 0:
            top_k_s_values = singular_values_T[:actual_k_components]
            q_j_val = top_k_s_values.sum() / actual_k_components
        else:
            q_j_val = torch.tensor(0.0, device=F_H.device)

        if q_j_val.isnan(): q_j_val = torch.tensor(0.0, device=F_H.device)

        return self.activation(q_j_val.unsqueeze(0).repeat(batch_size, 1)), T_matrix


class MCM_AFMC(nn.Module):
    def __init__(self, combined_feature_dim, max_items, num_semantic_masks_K, mapped_feature_dim_per_mask,
                 dcca_output_dim, k_principal_components, alpha=0.5, dropout_rate_finegrained=0.3,
                 dropout_rate_dcca=0.3):
        super().__init__()
        self.alpha = alpha
        self.fine_grained_module = OutfitAwareFineGrainedCompatibility(
            combined_feature_dim, num_semantic_masks_K, mapped_feature_dim_per_mask,
            dropout_rate=dropout_rate_finegrained
        )
        self.multi_view_module = MultiViewCorrelationModule(
            combined_feature_dim, max_items, dcca_output_dim, k_principal_components, dropout_rate=dropout_rate_dcca
        )

    def forward(self, item_features_cat, outfit_mask, outfit_labels=None):
        p_j = self.fine_grained_module(item_features_cat, outfit_mask)
        q_j, T_matrix_for_loss = self.multi_view_module(item_features_cat, outfit_mask)

        p_j = torch.clamp(p_j, 1e-7, 1.0 - 1e-7)
        q_j = torch.clamp(q_j, 1e-7, 1.0 - 1e-7)

        final_score_zeta_j = (1 - self.alpha) * p_j + self.alpha * q_j
        loss_com, loss_corr = None, None

        if outfit_labels is not None:
            if outfit_labels.device != p_j.device:
                outfit_labels = outfit_labels.to(p_j.device)

            loss_com = F.binary_cross_entropy(p_j, outfit_labels.float(), reduction='mean')

            try:
                singular_values_T = torch.linalg.svdvals(T_matrix_for_loss)

                k_for_loss = self.multi_view_module.k_principal_components
                d_out_for_loss = self.multi_view_module.dcca_output_dim

                actual_k_for_loss = min(k_for_loss, singular_values_T.numel(), d_out_for_loss)

                if actual_k_for_loss > 0:
                    loss_corr = -singular_values_T[:actual_k_for_loss].sum() / actual_k_for_loss
                else:
                    loss_corr = torch.tensor(0.0, device=p_j.device)

                if loss_corr.isnan():
                    loss_corr = torch.tensor(0.0, device=p_j.device)

            except RuntimeError as e:
                loss_corr = torch.tensor(0.0, device=p_j.device)

        return final_score_zeta_j, loss_com, loss_corr


class PolyvoreDataset(Dataset):
    def __init__(self, data_dir_root, compatibility_json_file_path, metadata_file_name, image_folder_relative_path,
                 max_items, transform=None, device='cpu', precomputed_features_path=None,
                 visual_feature_extractor=None, text_feature_extractor=None):

        t_init_start = time.time()

        self.data_dir_root = data_dir_root
        self.max_items = max_items
        self.transform = transform if transform else self._get_default_transform()
        self.device_for_dynamic_extraction = device
        self.precomputed_features = None

        self.VISUAL_FEATURE_DIM = 512
        if visual_feature_extractor and hasattr(visual_feature_extractor, 'output_dim'):
            self.VISUAL_FEATURE_DIM = visual_feature_extractor.output_dim

        self.TEXT_FEATURE_DIM = 768
        if text_feature_extractor and hasattr(text_feature_extractor, 'output_dim'):
            self.TEXT_FEATURE_DIM = text_feature_extractor.output_dim

        self.COMBINED_FEATURE_DIM = self.VISUAL_FEATURE_DIM + self.TEXT_FEATURE_DIM

        t_load_precomp_start = time.time()
        if precomputed_features_path and os.path.exists(precomputed_features_path):
            print(f"正在从 {precomputed_features_path} 加载预计算特征...")
            try:
                self.precomputed_features = torch.load(precomputed_features_path, map_location=torch.device('cpu'))
                print(
                    f"预计算特征已加载到 CPU。耗时 {time.time() - t_load_precomp_start:.2f}s。具有预计算特征的物品数量：{len(self.precomputed_features)}")
            except Exception as e:
                print(
                    f"加载预计算特征 '{precomputed_features_path}' 时出错：{e}。如果提供了提取器，将尝试动态提取。")
                self.precomputed_features = None
        else:
            print(
                f"未提供预计算特征路径 '{precomputed_features_path}' 或文件不存在。检查耗时 {time.time() - t_load_precomp_start:.2f}s。")

        self.do_dynamic_extraction = self.precomputed_features is None
        if self.do_dynamic_extraction:
            print("由于预计算特征不可用/未加载，将使用动态特征提取。")
            if visual_feature_extractor is None or text_feature_extractor is None:
                print("警告：至少一个特征提取器未完全可用于动态提取。")

            if visual_feature_extractor:
                self.visual_feature_extractor = visual_feature_extractor.to(self.device_for_dynamic_extraction).eval()
            else:
                self.visual_feature_extractor = None

            if text_feature_extractor and text_feature_extractor.tokenizer is not None and text_feature_extractor.bert is not None:
                self.text_feature_extractor = text_feature_extractor.to(self.device_for_dynamic_extraction).eval()
            else:
                self.text_feature_extractor = None
                if self.do_dynamic_extraction:
                    print(
                        "警告：文本特征提取器未正确初始化（可能由于网络/BERT模型加载问题）。动态文本提取可能失败或产生零特征。")
        else:
            self.visual_feature_extractor, self.text_feature_extractor = None, None
            print("使用预计算特征。Dataset 将不使用动态提取器。")

        t_load_meta_start = time.time()
        self.item_metadata = {}
        try:
            with open(os.path.join(self.data_dir_root, metadata_file_name), 'r', encoding='utf-8') as f:
                self.item_metadata = json.load(f)
            print(
                f"物品元数据已加载。耗时 {time.time() - t_load_meta_start:.2f}s。元数据条目数：{len(self.item_metadata)}")
        except FileNotFoundError:
            print(f"警告：元数据文件 {os.path.join(self.data_dir_root, metadata_file_name)} 未找到。")
        except json.JSONDecodeError:
            print(f"警告：解码元数据 JSON {metadata_file_name} 时出错。")

        self.outfits = []
        t_load_compat_start = time.time()
        print(f"正在从 {compatibility_json_file_path} 加载兼容性数据")
        try:
            with open(compatibility_json_file_path, 'r', encoding='utf-8') as f:
                compatibility_data = json.load(f)

            if not isinstance(compatibility_data, list):
                print(f"严重错误：来自 {compatibility_json_file_path} 的兼容性数据不是列表！")
                compatibility_data = []

            for outfit_entry in compatibility_data:
                label_str = outfit_entry.get("label")
                items_data_list = outfit_entry.get("items")

                if label_str is None or not isinstance(items_data_list, list) or not items_data_list:
                    continue

                current_outfit_item_ids = []
                current_outfit_item_texts_from_json = []

                for item_detail_dict in items_data_list:
                    if isinstance(item_detail_dict, dict) and "im" in item_detail_dict:
                        item_id = str(item_detail_dict["im"])
                        current_outfit_item_ids.append(item_id)
                        current_outfit_item_texts_from_json.append(item_detail_dict.get("text", ""))

                if current_outfit_item_ids:
                    self.outfits.append({'ids': current_outfit_item_ids,
                                         'texts_json': current_outfit_item_texts_from_json,
                                         'label': int(label_str)})
        except FileNotFoundError:
            print(f"严重错误：在 {compatibility_json_file_path} 未找到兼容性文件")
        except json.JSONDecodeError as e:
            print(f"严重错误：从 {compatibility_json_file_path} 解码兼容性 JSON 失败。错误：{e}")
        except Exception as e_compat:
            print(f"严重错误：加载兼容性数据时发生意外错误：{e_compat}")

        print(
            f"兼容性数据处理完成。找到 {len(self.outfits)} 个 outfits。耗时 {time.time() - t_load_compat_start:.2f}s")
        self.image_dir_abs_path = os.path.join(self.data_dir_root, image_folder_relative_path)

        if len(self.outfits) == 0:
            print("严重警告：未加载任何 outfits。请检查路径、JSON 结构和 ID 匹配逻辑。")
        print(f"PolyvoreDataset 初始化完成。总耗时：{time.time() - t_init_start:.2f}s")

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.outfits)

    def __getitem__(self, idx):
        outfit_data = self.outfits[idx]
        item_ids_in_outfit = outfit_data['ids']
        texts_from_json_for_outfit = outfit_data['texts_json']
        label = outfit_data['label']

        item_visual_features_list = []
        item_text_features_list_intermediate = []
        texts_for_dynamic_bert_batch = []
        item_indices_for_dynamic_text = []

        actual_items_processed_count = 0

        for i, item_id_str in enumerate(item_ids_in_outfit):
            if actual_items_processed_count >= self.max_items: break

            vis_feat, txt_feat_prelim = None, None

            text_to_use_for_bert = texts_from_json_for_outfit[i] if i < len(texts_from_json_for_outfit) and \
                                                                    texts_from_json_for_outfit[i] else ""
            if not text_to_use_for_bert and item_id_str in self.item_metadata:
                text_to_use_for_bert = self.item_metadata[item_id_str].get('name', "")

            if not self.do_dynamic_extraction and self.precomputed_features and item_id_str in self.precomputed_features:
                features = self.precomputed_features[item_id_str]
                vis_feat = features.get('visual')
                txt_feat_prelim = features.get('text')
            else:
                img_path = os.path.join(self.image_dir_abs_path, f"{item_id_str}.jpg")
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image)
                    if self.visual_feature_extractor:
                        with torch.no_grad():
                            vis_feat = self.visual_feature_extractor(
                                image_tensor.unsqueeze(0).to(self.device_for_dynamic_extraction)).squeeze(0).cpu()
                except Exception:
                    pass

                if self.do_dynamic_extraction and self.text_feature_extractor:
                    texts_for_dynamic_bert_batch.append(text_to_use_for_bert)
                    item_indices_for_dynamic_text.append(actual_items_processed_count)

            item_visual_features_list.append(vis_feat if vis_feat is not None else torch.zeros(self.VISUAL_FEATURE_DIM))
            item_text_features_list_intermediate.append(txt_feat_prelim if txt_feat_prelim is not None else None)
            actual_items_processed_count += 1

        if self.do_dynamic_extraction and self.text_feature_extractor and texts_for_dynamic_bert_batch:
            if texts_for_dynamic_bert_batch:
                with torch.no_grad():
                    batch_text_feats_dyn = self.text_feature_extractor(texts_for_dynamic_bert_batch,
                                                                       self.device_for_dynamic_extraction).cpu()
                for i, original_idx in enumerate(item_indices_for_dynamic_text):
                    if original_idx < len(item_text_features_list_intermediate):
                        item_text_features_list_intermediate[original_idx] = batch_text_feats_dyn[i]

        final_item_text_features = [feat if feat is not None else torch.zeros(self.TEXT_FEATURE_DIM) for feat in
                                    item_text_features_list_intermediate]

        combined_features_list = []
        num_items_to_combine = min(len(item_visual_features_list), len(final_item_text_features))

        for i in range(num_items_to_combine):
            vis_i = item_visual_features_list[i]
            txt_i = final_item_text_features[i]
            combined_features_list.append(torch.cat((vis_i, txt_i), dim=0))

        if not combined_features_list:
            features_tensor = torch.empty(0, self.COMBINED_FEATURE_DIM)
        else:
            features_tensor = torch.stack(combined_features_list)

        num_valid_items_in_tensor = features_tensor.shape[0]
        return features_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(num_valid_items_in_tensor,
                                                                                    dtype=torch.long)


def collate_fn_polyvore(batch, max_items_global, feature_dim_global):
    features_list, labels_list, num_items_actual_list = zip(*batch)

    padded_features_batch = []
    actual_num_items_for_mask = [min(n.item(), max_items_global) for n in num_items_actual_list]

    for i, feat_tensor in enumerate(features_list):
        num_actual_in_tensor = feat_tensor.shape[0]
        current_items_to_pad_from_tensor = min(num_actual_in_tensor, max_items_global)

        padded_tensor_for_item = torch.zeros(max_items_global, feature_dim_global, dtype=torch.float)

        if current_items_to_pad_from_tensor > 0:
            if feat_tensor.shape[1] != feature_dim_global:
                pass
            else:
                valid_part = feat_tensor[:current_items_to_pad_from_tensor, :]
                padded_tensor_for_item[:current_items_to_pad_from_tensor, :] = valid_part

        padded_features_batch.append(padded_tensor_for_item)

    final_features_batch = torch.stack(padded_features_batch)
    final_labels_batch = torch.stack(labels_list).unsqueeze(1)
    final_outfit_mask = get_outfit_mask(torch.tensor(actual_num_items_for_mask, dtype=torch.long), max_items_global)

    return final_features_batch, final_labels_batch, final_outfit_mask


def precompute_and_save_features(data_root_dir_for_meta_img, metadata_file_name, image_folder_relative_path,
                                 output_path,
                                 visual_extractor, text_extractor, transform, device,
                                 VISUAL_FEATURE_DIM_CONST, TEXT_FEATURE_DIM_CONST):
    t_precompute_start_total = time.time()
    print("开始特征预计算...")

    if text_extractor is not None and (text_extractor.tokenizer is None or text_extractor.bert is None):
        print("*" * 80)
        print("严重警告：文本特征提取器 (TextFeatureExtractor) 未能成功加载其BERT模型或tokenizer。")
        print("这通常是由于网络问题无法连接到Hugging Face Hub，或者本地缓存的模型损坏/不完整。")
        print("预计算将继续，但所有文本特征都将被设置为零向量。")
        print("这将严重影响模型性能。请解决BERT加载问题并强制重新计算特征。")
        print("*" * 80)
    elif text_extractor is None:
        print("*" * 80)
        print("警告：未提供文本特征提取器。预计算的文本特征将为零向量。")
        print("*" * 80)

    item_metadata_path = os.path.join(data_root_dir_for_meta_img, metadata_file_name)
    item_metadata_dict = {}
    try:
        with open(item_metadata_path, 'r', encoding='utf-8') as f:
            item_metadata_dict = json.load(f)
    except FileNotFoundError:
        print(f"警告：预计算期间未找到元数据文件 {item_metadata_path}。")
    except json.JSONDecodeError:
        print(f"警告：预计算期间解码元数据 JSON {metadata_file_name} 时出错。")

    all_item_ids_from_metadata = list(item_metadata_dict.keys())
    if not all_item_ids_from_metadata:
        print("严重错误：元数据中未找到物品 ID。预计算无法进行。")
        torch.save({}, output_path)
        return {}

    precomputed_features = {}
    if visual_extractor: visual_extractor.eval().to(device)
    if text_extractor: text_extractor.eval().to(device)

    batch_size_precompute = 64
    num_batches = (len(all_item_ids_from_metadata) + batch_size_precompute - 1) // batch_size_precompute
    processed_count_total = 0
    image_not_found_count = 0

    for i in range(num_batches):
        t_batch_start = time.time()
        current_batch_item_ids_meta = all_item_ids_from_metadata[
                                      i * batch_size_precompute: (i + 1) * batch_size_precompute]

        batch_image_tensors_for_resnet = []
        batch_texts_for_bert = []
        item_ids_for_this_batch_output = []

        for item_id_str in current_batch_item_ids_meta:
            img_path = os.path.join(data_root_dir_for_meta_img, image_folder_relative_path, f"{item_id_str}.jpg")
            meta_entry = item_metadata_dict.get(str(item_id_str), {})
            text_desc = meta_entry.get('name', "")

            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_image_tensors_for_resnet.append(image_tensor)
                batch_texts_for_bert.append(text_desc)
                item_ids_for_this_batch_output.append(item_id_str)
            except FileNotFoundError:
                image_not_found_count += 1
                continue
            except Exception as e_img_load:
                print(f"为物品 {item_id_str} 加载图片 {img_path} 时出错: {e_img_load}")
                continue

        if not item_ids_for_this_batch_output: continue

        vis_features_batch = torch.zeros(len(item_ids_for_this_batch_output), VISUAL_FEATURE_DIM_CONST, device='cpu')
        if batch_image_tensors_for_resnet and visual_extractor:
            try:
                stacked_images = torch.stack(batch_image_tensors_for_resnet).to(device)
                with torch.no_grad():
                    vis_features_batch = visual_extractor(stacked_images).cpu()
            except Exception as e_vis:
                print(f"为批次 {i + 1} 提取视觉特征时出错: {e_vis}")

        txt_features_batch = torch.zeros(len(item_ids_for_this_batch_output), TEXT_FEATURE_DIM_CONST, device='cpu')
        if batch_texts_for_bert and text_extractor and text_extractor.tokenizer and text_extractor.bert:
            try:
                with torch.no_grad():
                    txt_features_batch = text_extractor(batch_texts_for_bert, device).cpu()
            except Exception as e_txt:
                print(f"为批次 {i + 1} 提取文本特征时出错: {e_txt}")

        for j, item_id_to_save in enumerate(item_ids_for_this_batch_output):
            precomputed_features[item_id_to_save] = {
                'visual': vis_features_batch[j].clone(),
                'text': txt_features_batch[j].clone()
            }

        processed_count_total += len(item_ids_for_this_batch_output)
        if (i + 1) % 10 == 0 or i == num_batches - 1:
            print(
                f"预计算批次 {i + 1}/{num_batches}。批次耗时: {time.time() - t_batch_start:.2f}s。"
                f"总处理数: {processed_count_total}。图片未找到: {image_not_found_count}")

    if not precomputed_features:
        print("警告：未预计算任何特征。正在保存空字典。")
        torch.save({}, output_path)
    else:
        torch.save(precomputed_features, output_path)
        print(f"预计算特征已保存到 {output_path}。物品数量: {len(precomputed_features)}")

    print(
        f"总预计算耗时: {time.time() - t_precompute_start_total:.2f}s。总图片未找到数: {image_not_found_count}")
    return precomputed_features


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for item_feats, labels, outfit_m in dataloader:
            item_feats = item_feats.to(device)
            labels_on_device = labels.to(device)
            outfit_m = outfit_m.to(device)

            final_scores, _, _ = model(item_feats, outfit_m, labels_on_device.float())

            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(final_scores.cpu().numpy().flatten())

    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        try:
            all_predictions_np = np.array(all_predictions)
            if not np.all(np.isfinite(all_predictions_np)):
                all_predictions_np = np.nan_to_num(all_predictions_np, nan=0.5, posinf=1.0, neginf=0.0)
                all_predictions_np = np.clip(all_predictions_np, 0.0, 1.0)

            auc_score = roc_auc_score(all_labels, all_predictions_np)
            return auc_score
        except ValueError as e:
            print(f"警告：无法计算 AUC。错误：{e}。标签唯一值: {np.unique(all_labels)}。预测样本: {all_predictions_np[:5]}")
            return None
    else:
        print(
            f"警告：验证集中数据不足 ({len(all_labels)} 个样本) 或只有一个类别 ({np.unique(all_labels)})，无法计算 AUC。")
        return None


if __name__ == '__main__':
    VISUAL_FEATURE_DIM = 512
    TEXT_FEATURE_DIM = 768
    COMBINED_FEATURE_DIM = VISUAL_FEATURE_DIM + TEXT_FEATURE_DIM
    MAX_ITEMS_IN_OUTFIT = 8
    NUM_SEMANTIC_MASKS_K = 5
    MAPPED_FEATURE_DIM_PER_MASK = 64

    BATCH_SIZE = 128
    DCCA_OUTPUT_DIM = 24
    K_PRINCIPAL_COMPONENTS_DCCA = min(10, DCCA_OUTPUT_DIM)

    ALPHA_BALANCE = 0.5
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    REG_LAMBDA = 1e-5
    DROPOUT_RATE = 0.3

    DATA_ROOT_DIR = 'polyvore_outfits'
    IMAGE_FOLDER_RELATIVE_PATH = 'images'
    CURRENT_DATA_SPLIT_DIR_NAME = 'disjoint'
    CURRENT_DATA_SPLIT_PATH = os.path.join(DATA_ROOT_DIR, CURRENT_DATA_SPLIT_DIR_NAME)
    METADATA_FILENAME = 'polyvore_item_metadata.json'
    PRECOMPUTED_FEATURES_FILE = os.path.join(DATA_ROOT_DIR,
                                             f'precomputed_features_polyvore_{CURRENT_DATA_SPLIT_DIR_NAME}_v2.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_feat_extractor_instance = None
    text_feat_extractor_instance = None

    force_recompute = False

    print("正在初始化特征提取器 (可能会尝试 HuggingFace 连接)...")
    image_feat_extractor_instance = ImageFeatureExtractor(VISUAL_FEATURE_DIM)
    text_feat_extractor_instance = TextFeatureExtractor(TEXT_FEATURE_DIM)
    print("特征提取器已初始化。")

    if force_recompute or not os.path.exists(PRECOMPUTED_FEATURES_FILE):
        print(
            f"预计算特征文件 '{PRECOMPUTED_FEATURES_FILE}' 未找到或 force_recompute=True。开始预计算...")
        precompute_and_save_features(
            DATA_ROOT_DIR,
            METADATA_FILENAME,
            IMAGE_FOLDER_RELATIVE_PATH,
            PRECOMPUTED_FEATURES_FILE,
            image_feat_extractor_instance,
            text_feat_extractor_instance,
            default_transform,
            device, VISUAL_FEATURE_DIM, TEXT_FEATURE_DIM
        )
    else:
        print(f"在 {PRECOMPUTED_FEATURES_FILE} 找到预计算特征")

    model = MCM_AFMC(
        COMBINED_FEATURE_DIM, MAX_ITEMS_IN_OUTFIT, NUM_SEMANTIC_MASKS_K,
        MAPPED_FEATURE_DIM_PER_MASK, DCCA_OUTPUT_DIM, K_PRINCIPAL_COMPONENTS_DCCA,
        alpha=ALPHA_BALANCE,
        dropout_rate_finegrained=DROPOUT_RATE,
        dropout_rate_dcca=DROPOUT_RATE
    ).to(device)

    train_compat_file_abs_path = os.path.join(CURRENT_DATA_SPLIT_PATH, 'compatibility_train_new.json')
    print(f"尝试从以下位置加载训练兼容性数据: {train_compat_file_abs_path}")
    if not os.path.exists(train_compat_file_abs_path):
        print(f"错误：未找到训练兼容性文件: {train_compat_file_abs_path}")
        exit(1)

    train_dataset = PolyvoreDataset(
        data_dir_root=DATA_ROOT_DIR,
        compatibility_json_file_path=train_compat_file_abs_path,
        metadata_file_name=METADATA_FILENAME,
        image_folder_relative_path=IMAGE_FOLDER_RELATIVE_PATH,
        max_items=MAX_ITEMS_IN_OUTFIT,
        transform=default_transform,
        device=device,
        precomputed_features_path=PRECOMPUTED_FEATURES_FILE,
        visual_feature_extractor=image_feat_extractor_instance,
        text_feature_extractor=text_feat_extractor_instance
    )
    if len(train_dataset) == 0: print("错误：PolyvoreDataset 初始化后训练数据集为空。"); exit(1)

    collate_with_params = partial(collate_fn_polyvore,
                                  max_items_global=MAX_ITEMS_IN_OUTFIT,
                                  feature_dim_global=COMBINED_FEATURE_DIM)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_params, num_workers=0,
                              pin_memory=True if device.type == 'cuda' else False,
                              persistent_workers=False)

    valid_loader = None
    valid_compat_file_abs_path = os.path.join(CURRENT_DATA_SPLIT_PATH, 'compatibility_valid_new.json')
    print(f"尝试从以下位置加载验证兼容性数据: {valid_compat_file_abs_path}")
    if not os.path.exists(valid_compat_file_abs_path):
        print(f"警告：在 {valid_compat_file_abs_path} 未找到验证兼容性文件。将不计算 AUC。")
    else:
        valid_dataset = PolyvoreDataset(
            data_dir_root=DATA_ROOT_DIR,
            compatibility_json_file_path=valid_compat_file_abs_path,
            metadata_file_name=METADATA_FILENAME,
            image_folder_relative_path=IMAGE_FOLDER_RELATIVE_PATH,
            max_items=MAX_ITEMS_IN_OUTFIT,
            transform=default_transform,
            device=device,
            precomputed_features_path=PRECOMPUTED_FEATURES_FILE,
            visual_feature_extractor=image_feat_extractor_instance,
            text_feature_extractor=text_feat_extractor_instance
        )
        if len(valid_dataset) > 0:
            valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=collate_with_params, num_workers=0,
                                      pin_memory=True if device.type == 'cuda' else False,
                                      persistent_workers=False)
        else:
            print("警告：验证数据集为空。将不计算 AUC。")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=REG_LAMBDA)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True, min_lr=1e-7)

    print("开始训练循环...")
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch, total_loss_com_epoch, total_loss_corr_epoch = 0.0, 0.0, 0.0
        num_batches_processed = 0
        epoch_start_time = time.time()

        for batch_idx, (item_feats, labels, outfit_m) in enumerate(train_loader):
            batch_start_time = time.time()
            item_feats = item_feats.to(device, non_blocking=True if device.type == 'cuda' else False)
            labels = labels.to(device, non_blocking=True if device.type == 'cuda' else False)
            outfit_m = outfit_m.to(device, non_blocking=True if device.type == 'cuda' else False)

            if item_feats.size(0) < 2:
                continue
            if item_feats.isnan().any():
                print(f"警告：跳过训练批次 {batch_idx + 1} 因为 item_feats 中存在 NaN。")
                continue

            optimizer.zero_grad(set_to_none=True)

            final_scores, loss_com, loss_corr = model(item_feats, outfit_m, labels.float())

            if loss_com is None or loss_corr is None or loss_com.isnan() or loss_corr.isnan():
                print(f"警告：跳过训练批次 {batch_idx + 1} 因为损失为 NaN (Com: {loss_com}, Corr: {loss_corr})。")
                continue

            loss_com_val = loss_com.item() if torch.is_tensor(loss_com) else loss_com
            loss_corr_val = loss_corr.item() if torch.is_tensor(loss_corr) else loss_corr

            total_loss = loss_com + loss_corr

            if total_loss.isnan():
                print(f"警告：跳过训练批次 {batch_idx + 1} 因为总损失为 NaN。")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += total_loss.item()
            total_loss_com_epoch += loss_com_val
            total_loss_corr_epoch += loss_corr_val
            num_batches_processed += 1

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch {epoch + 1}/{EPOCHS}, 批次 {batch_idx + 1}/{len(train_loader)}, "
                      f"学习率: {optimizer.param_groups[0]['lr']:.2e}, "
                      f"损失: {total_loss.item():.4f} (Com: {loss_com_val:.4f}, Corr: {loss_corr_val:.4f}), "
                      f"批次耗时: {time.time() - batch_start_time:.2f}s")

        epoch_duration = time.time() - epoch_start_time
        if num_batches_processed > 0:
            avg_loss = total_loss_epoch / num_batches_processed
            avg_com_loss = total_loss_com_epoch / num_batches_processed
            avg_corr_loss = total_loss_corr_epoch / num_batches_processed
            print_msg = (f"Epoch {epoch + 1} 完成。平均损失: {avg_loss:.4f} "
                         f"(Com: {avg_com_loss:.4f}, Corr: {avg_corr_loss:.4f})。 "
                         f"耗时: {epoch_duration:.2f}s")

            current_auc_for_scheduler = 0.0
            if valid_loader:
                auc = evaluate_model(model, valid_loader, device)
                if auc is not None:
                    current_auc_for_scheduler = auc
                    print_msg += f" | 验证集 AUC: {auc:.4f}"
                    if auc > best_val_auc:
                        best_val_auc = auc
                scheduler.step(current_auc_for_scheduler)
            else:
                scheduler.step(avg_loss)

            print(print_msg)
        else:
            print(f"Epoch {epoch + 1} 没有成功处理任何批次。耗时: {epoch_duration:.2f}s")
            if valid_loader:
                scheduler.step(0)
            else:
                scheduler.step(float('inf'))

    print(f"训练完成。最佳验证集 AUC: {best_val_auc:.4f}")