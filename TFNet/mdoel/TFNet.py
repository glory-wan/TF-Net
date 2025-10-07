import torch
import torch.nn as nn
import torch.nn.functional as F

# Import MobileOne components with fallback for different import contexts
try:
    from .mobileone import reparameterize_model, MobileOneBlock
except ImportError:
    from mobileone import reparameterize_model, MobileOneBlock


class TFNet(nn.Module):
    """
    TF-Net: Main segmentation network for Tear Film Break-Up (TFBU) analysis.

    Features:
    - MobileOne-mini backbone with re-parameterization for efficiency
    - Pyramid Pooling Module (PPM) for multi-scale context aggregation
    - Skip connections for boundary refinement
    - Designed for real-time clinical deployment

    Args:
        num_classes (int): Number of output classes. Default 1 for binary segmentation.
        width (float): Width multiplier for network channels. Default 1.0.
        use_se (bool): Whether to use Squeeze-and-Excitation blocks. Default False.
        inference_mode (bool): If True, uses re-parameterized model for inference.
        base_channels (list): Base channel dimensions for each stage.
        use_ppm (bool): Whether to use Pyramid Pooling Module. Default True.
        use_skip_connections (bool): Whether to use skip connections. Default True.
    """

    def __init__(self, num_classes=1, width=1.0, use_se=False, inference_mode=False,
                 base_channels=[64, 128, 256, 512], use_ppm=True, use_skip_connections=True):
        super().__init__()
        self.num_classes = num_classes
        self.width = width
        self.use_se = use_se
        self.inference_mode = inference_mode
        self.use_ppm = use_ppm
        self.use_skip_connections = use_skip_connections
        self.channels = [int(c * width) for c in base_channels]

        # Encoder backbone: MobileOne-mini for feature extraction
        self.encoder = MobileOneMini(
            width=width,
            use_se=use_se,
            base_channels=base_channels,
            inference_mode=inference_mode
        )

        # Pyramid Pooling Module for multi-scale context aggregation
        if use_ppm:
            ppm_out_channels = max(64, self.channels[-1] // 4)
            self.ppm = PyramidPoolingModule(
                in_channels=self.channels[-1],
                out_channels=ppm_out_channels,
                sizes=[1, 2, 3, 6],  # Multi-scale pooling sizes
                use_se=use_se,
                inference_mode=inference_mode
            )
            decoder_input_channels = ppm_out_channels
        else:
            # Fallback: use encoder's final output directly without PPM
            decoder_input_channels = self.channels[-1]

        # Decoder channel configuration
        decoder_channels = [
            decoder_input_channels // 2,
            decoder_input_channels // 4,
            decoder_input_channels // 8
        ]

        # Decoder for feature map upsampling and refinement
        self.decoder = Decoder(
            encoder_channels=self.channels,
            decoder_channels=decoder_channels,
            decoder_input_channels=decoder_input_channels,
            use_skip_connections=use_skip_connections,
            use_se=use_se,
            inference_mode=inference_mode
        )

        # Final convolution layer for output prediction
        self.final_conv = nn.Conv2d(
            decoder_channels[-1], num_classes,
            kernel_size=1, bias=True  # 1x1 convolution for classification
        )

    def forward(self, x):
        """
        Forward pass of TF-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Output segmentation map of shape (B, num_classes, H, W)
        """
        # Extract multi-scale features from encoder
        features = self.encoder(x)

        # Apply Pyramid Pooling Module if enabled
        if self.use_ppm:
            decoder_input = self.ppm(features[-1])  # Process deepest features
        else:
            decoder_input = features[-1]  # Use raw deepest features

        # Decode features with skip connections
        x = self.decoder(features, decoder_input)

        # Final prediction and upsampling to match input resolution
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Apply appropriate activation based on number of classes
        if self.num_classes == 1:
            # Binary segmentation: sigmoid for single-class probability
            return torch.sigmoid(x)
        else:
            # Multi-class segmentation: softmax for class probabilities
            return F.softmax(x, dim=1)


class MobileOneMini(nn.Module):
    """
    Lightweight MobileOne-mini backbone for feature extraction.

    Optimized for tear film image analysis with reduced computational complexity
    while maintaining representational capacity.

    Args:
        width (float): Width multiplier for channel dimensions.
        use_se (bool): Whether to use Squeeze-and-Excitation attention.
        inference_mode (bool): Inference-optimized reparameterization mode.
        num_blocks_per_stage (list): Number of blocks in each stage.
        base_channels (list): Base channel dimensions for each stage.
    """

    def __init__(self, width=1.0, use_se=False, inference_mode=False,
                 num_blocks_per_stage=[2, 3, 4, 3],
                 base_channels=[64, 128, 256, 512]):
        super().__init__()
        self.use_se = use_se
        self.inference_mode = inference_mode

        # Scale channels according to width multiplier
        channels = [int(c * width) for c in base_channels]

        # Stem: initial feature extraction with stride 2 for resolution reduction
        self.stem = nn.Sequential(
            MobileOneBlock(3, channels[0], 3, stride=2, padding=1,
                           use_se=use_se, inference_mode=inference_mode),
            MobileOneBlock(channels[0], channels[0], 3, stride=1, padding=1,
                           use_se=use_se, inference_mode=inference_mode)
        )

        # Four downsampling stages with progressive channel increase
        self.stage1 = self._make_stage(channels[0], channels[0], num_blocks=num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], num_blocks=num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], num_blocks=num_blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], num_blocks=num_blocks_per_stage[3], stride=2)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Construct a network stage with specified parameters.

        Args:
            in_channels (int): Input channel dimension.
            out_channels (int): Output channel dimension.
            num_blocks (int): Number of MobileOne blocks in the stage.
            stride (int): Stride for the first block (controls downsampling).

        Returns:
            nn.Sequential: Sequential container of MobileOne blocks.
        """
        layers = []
        # First block may use stride for downsampling
        layers.append(MobileOneBlock(
            in_channels, out_channels, 3, stride=stride, padding=1,
            use_se=self.use_se, inference_mode=self.inference_mode
        ))

        # Subsequent blocks maintain resolution
        for _ in range(1, num_blocks):
            layers.append(MobileOneBlock(
                out_channels, out_channels, 3, stride=1, padding=1,
                use_se=self.use_se, inference_mode=self.inference_mode
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through MobileOne-mini backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            list: List of multi-scale feature maps at different resolutions:
                [1/2, 1/2, 1/4, 1/8, 1/16] of input resolution
        """
        features = []

        x = self.stem(x)
        features.append(x)  # 1/2 resolution

        x = self.stage1(x)
        features.append(x)  # 1/2 resolution

        x = self.stage2(x)
        features.append(x)  # 1/4 resolution

        x = self.stage3(x)
        features.append(x)  # 1/8 resolution

        x = self.stage4(x)
        features.append(x)  # 1/16 resolution

        return features


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) for capturing multi-scale contextual information.

    Inspired by PSPNet, this module aggregates context at multiple scales
    to handle variable-sized TFBU regions.

    Args:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimension for each pyramid branch.
        sizes (list): List of pooling sizes for pyramid levels.
        use_se (bool): Whether to use SE blocks in MobileOne.
        inference_mode (bool): Inference-optimized mode.
    """

    def __init__(self, in_channels, out_channels, sizes=[1, 2, 3, 6],
                 use_se=False, inference_mode=False):
        super().__init__()
        self.out_channels = out_channels

        # Multiple pyramid branches with different pooling scales
        self.pyramid_branches = nn.ModuleList()
        for size in sizes:
            self.pyramid_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),  # Adaptive pooling to fixed size
                MobileOneBlock(in_channels, out_channels, 1, stride=1, padding=0,
                               use_se=use_se, inference_mode=inference_mode)
            ))

        # Main branch processing original features
        self.main_branch = MobileOneBlock(in_channels, out_channels, 1, stride=1, padding=0,
                                          use_se=use_se, inference_mode=inference_mode)

        # Fusion of all pyramid features
        fusion_in_channels = out_channels * (len(sizes) + 1)  # +1 for main branch
        self.fusion = MobileOneBlock(
            fusion_in_channels, out_channels, 1,
            stride=1, padding=0, use_se=use_se, inference_mode=inference_mode
        )

    def forward(self, x):
        """
        Forward pass through Pyramid Pooling Module.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            torch.Tensor: Fused multi-scale features of shape (B, out_channels, H, W)
        """
        h, w = x.shape[2], x.shape[3]  # Original spatial dimensions

        # Process through main branch
        main = self.main_branch(x)

        # Process through pyramid branches and upsample to original size
        pyramid_features = [main]
        for branch in self.pyramid_branches:
            pyramid_feat = branch(x)
            pyramid_feat = F.interpolate(pyramid_feat, size=(h, w),
                                         mode='bilinear', align_corners=False)
            pyramid_features.append(pyramid_feat)

        # Concatenate and fuse all features
        x = torch.cat(pyramid_features, dim=1)
        x = self.fusion(x)

        return x


class Decoder(nn.Module):
    """
    Decoder module for feature map upsampling and refinement.

    Progressively recovers spatial resolution while incorporating
    skip connections from encoder for boundary preservation.

    Args:
        encoder_channels (list): Channel dimensions from encoder stages.
        decoder_channels (list): Target channel dimensions for decoder stages.
        decoder_input_channels (int): Input channels to decoder.
        use_skip_connections (bool): Whether to use encoder skip connections.
        use_se (bool): Whether to use SE blocks.
        inference_mode (bool): Inference-optimized mode.
    """

    def __init__(self, encoder_channels, decoder_channels, decoder_input_channels,
                 use_skip_connections=True, use_se=False, inference_mode=False):
        super().__init__()
        self.use_se = use_se
        self.inference_mode = inference_mode
        self.use_skip_connections = use_skip_connections

        # Three upsampling stages with optional skip connections
        self.up4 = UpsampleBlock(
            in_channels=decoder_input_channels,
            skip_channels=encoder_channels[-2] if use_skip_connections else 0,
            out_channels=decoder_channels[0],
            use_skip_connections=use_skip_connections,
            use_se=use_se,
            inference_mode=inference_mode
        )

        self.up3 = UpsampleBlock(
            in_channels=decoder_channels[0],
            skip_channels=encoder_channels[-3] if use_skip_connections else 0,
            out_channels=decoder_channels[1],
            use_skip_connections=use_skip_connections,
            use_se=use_se,
            inference_mode=inference_mode
        )

        self.up2 = UpsampleBlock(
            in_channels=decoder_channels[1],
            skip_channels=encoder_channels[-4] if use_skip_connections else 0,
            out_channels=decoder_channels[2],
            use_skip_connections=use_skip_connections,
            use_se=use_se,
            inference_mode=inference_mode
        )

    def forward(self, encoder_features, decoder_input):
        """
        Forward pass through decoder.

        Args:
            encoder_features (list): Multi-scale features from encoder.
            decoder_input (torch.Tensor): Initial decoder input features.

        Returns:
            torch.Tensor: Decoded features at higher resolution.
        """
        x = decoder_input

        # Progressive upsampling with optional skip connections
        if self.use_skip_connections:
            x = self.up4(x, encoder_features[3])  # 1/8 → 1/4
            x = self.up3(x, encoder_features[2])  # 1/4 → 1/2
            x = self.up2(x, encoder_features[1])  # 1/2 → 1/1
        else:
            # Without skip connections: pure upsampling pathway
            x = self.up4(x, None)
            x = self.up3(x, None)
            x = self.up2(x, None)

        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block with optional skip connection fusion.

    Each block performs 2x upsampling and optionally fuses with skip connections
    from encoder at corresponding resolution.

    Args:
        in_channels (int): Input channel dimension.
        skip_channels (int): Skip connection channel dimension (0 if disabled).
        out_channels (int): Output channel dimension.
        use_skip_connections (bool): Whether to use skip connections.
        use_se (bool): Whether to use SE blocks.
        inference_mode (bool): Inference-optimized mode.
    """

    def __init__(self, in_channels, skip_channels, out_channels,
                 use_skip_connections=True, use_se=False, inference_mode=False):
        super().__init__()
        self.use_skip_connections = use_skip_connections

        # Bilinear upsampling for resolution recovery
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Calculate fusion input channels (with or without skip connections)
        fusion_input_channels = in_channels + skip_channels if use_skip_connections else in_channels

        # Feature fusion with two MobileOne blocks
        self.fusion = nn.Sequential(
            MobileOneBlock(fusion_input_channels, out_channels, 3,
                           stride=1, padding=1, use_se=use_se, inference_mode=inference_mode),
            MobileOneBlock(out_channels, out_channels, 3,
                           stride=1, padding=1, use_se=use_se, inference_mode=inference_mode)
        )

    def forward(self, x, skip):
        """
        Forward pass through upsampling block.

        Args:
            x (torch.Tensor): Input features from previous stage.
            skip (torch.Tensor): Skip connection features from encoder.

        Returns:
            torch.Tensor: Upsampled and fused features.
        """
        # Upsample input features
        x = self.upsample(x)

        # Fuse with skip connection if available and enabled
        if self.use_skip_connections and skip is not None:
            # Ensure spatial dimension matching
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)

        # Process fused features
        x = self.fusion(x)

        return x


def create_tfnet_variants(num_classes=1, width=1.0, use_se=False, inference_mode=False):
    """
    Factory function to create different TF-Net variants for ablation studies.

    Returns four model configurations:
    - full: Complete TF-Net with PPM and skip connections
    - no_ppm: Without Pyramid Pooling Module
    - no_skip: Without skip connections
    - no_ppm_no_skip: Minimal version without both PPM and skip connections

    Args:
        num_classes (int): Number of output classes.
        width (float): Width multiplier for channels.
        use_se (bool): Whether to use SE blocks.
        inference_mode (bool): Inference-optimized mode.

    Returns:
        dict: Dictionary containing all model variants.
    """

    # Complete TF-Net with all components
    full_model = TFNet(
        num_classes=num_classes,
        width=width,
        use_se=use_se,
        inference_mode=inference_mode,
        use_ppm=True,
        use_skip_connections=True
    )

    # Without Pyramid Pooling Module
    no_ppm_model = TFNet(
        num_classes=num_classes,
        width=width,
        use_se=use_se,
        inference_mode=inference_mode,
        use_ppm=False,
        use_skip_connections=True
    )

    # Without skip connections
    no_skip_model = TFNet(
        num_classes=num_classes,
        width=width,
        use_se=use_se,
        inference_mode=inference_mode,
        use_ppm=True,
        use_skip_connections=False
    )

    # Minimal version without both key components
    no_ppm_no_skip_model = TFNet(
        num_classes=num_classes,
        width=width,
        use_se=use_se,
        inference_mode=inference_mode,
        use_ppm=False,
        use_skip_connections=False
    )

    return {
        'full': full_model,
        'no_ppm': no_ppm_model,
        'no_skip': no_skip_model,
        'no_ppm_no_skip': no_ppm_no_skip_model
    }