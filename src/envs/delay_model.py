"""
FL 时延模型
T_path: 单条路径的传输时延
T_up:   上传阶段时延 = max(所有GBS的路径时延)  -- 掉队者效应
T_down: 分发阶段时延 = avg(所有GBS的路径时延)
T_round = T_up + T_agg + T_down
"""
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DelayConfig:
    model_size: float = 10.0    # FL模型大小 (Mb)
    packet_size: float = 1.0    # 数据包大小 (Mb)
    hop_delay: float = 0.001    # 每跳固定处理时延 (s)
    t_agg: float = 0.5          # 聚合时延 (s)
    min_bandwidth: float = 0.1  # 避免除零的最小带宽 (Mbps)


class DelayModel:
    def __init__(self, cfg: DelayConfig = None):
        self.cfg = cfg or DelayConfig()

    def path_delay(self, path_nodes: List[int], topo) -> float:
        """
        计算单条路径的传输时延 (s)
        T_path = sum_{l in path} (M + L) / b_l  +  hop_delay * len(path)
        其中 M=model_size, L=packet_size, b_l=链路可用带宽
        """
        if len(path_nodes) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            bw = max(topo.available_bandwidth(u, v), self.cfg.min_bandwidth)
            total += (self.cfg.model_size + self.cfg.packet_size) / bw
        total += self.cfg.hop_delay * (len(path_nodes) - 1)
        return total

    def upload_delay(self, paths: Dict[int, List[int]], topo) -> Tuple[float, Dict[int, float]]:
        """
        T_up = max over all GBS of T_path(upload)
        返回 (T_up, per_gbs_delay_dict)
        """
        per_gbs = {}
        for gbs_id, path in paths.items():
            per_gbs[gbs_id] = self.path_delay(path, topo)
        t_up = max(per_gbs.values()) if per_gbs else 0.0
        return t_up, per_gbs

    def download_delay(self, paths: Dict[int, List[int]], topo) -> Tuple[float, Dict[int, float]]:
        """
        T_down = avg over all GBS of T_path(download)
        返回 (T_down, per_gbs_delay_dict)
        """
        per_gbs = {}
        for gbs_id, path in paths.items():
            per_gbs[gbs_id] = self.path_delay(path, topo)
        t_down = sum(per_gbs.values()) / len(per_gbs) if per_gbs else 0.0
        return t_down, per_gbs

    def round_delay(self, upload_paths: Dict[int, List[int]], topo,
                    download_paths: Dict[int, List[int]] = None) -> dict:
        """计算完整 FL round 时延"""
        t_up, up_per_gbs = self.upload_delay(upload_paths, topo)
        t_agg = self.cfg.t_agg
        t_down, down_per_gbs = 0.0, {}
        if download_paths:
            t_down, down_per_gbs = self.download_delay(download_paths, topo)
        return {
            "T_up": t_up,
            "T_agg": t_agg,
            "T_down": t_down,
            "T_round": t_up + t_agg + t_down,
            "up_per_gbs": up_per_gbs,
            "down_per_gbs": down_per_gbs,
        }

    def normalize_delay(self, delay: float, max_delay: float = 20.0) -> float:
        return min(delay / max_delay, 1.0)
