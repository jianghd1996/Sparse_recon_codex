"""LivePhoto diffusion with the native Wan VAE on both encode and decode."""

from .pipeline_wan2_2_fun_control_infer_wjx_livephoto_light import (
    Wan2_2LivePhotoPipeline_infer as _LivePhotoPipeline,
)
from .standard_wan_vae import StandardWanVaeCodec


class Wan2_2LivePhotoPipeline_infer(StandardWanVaeCodec, _LivePhotoPipeline):
    pass
