"""
fpn_pretrained.py - Contains a FPN based on a modified version of
EfficientNet-PyTorch built by lukemelas (github username) on the 
Github repo: https://github.com/lukemelas/EfficientNet-PyTorch 

My modified version DOES NOT use Squeeze-Excitation (SE) modules
as "SE connections tend to suppress localization of features in
favour of contextual elements."

Therefore, this folder REQUIRES to use the EfficientNet-PyTorch repo.
"""


import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

# You can get the repo needed below by runnning
# git submodule update --init --recursive
from ..third_party.efficient_net_pytorch.efficientnet_pytorch.utils import *
from .utilities import (
    DepthSeparableConv2d,
    MobileInvertedBottleneck,
    conv_1x1_bn,
)

VALID_MODELS = (
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "efficientnet-b8",
    # Support the construction of 'efficientnet-l2' without pretrained weights
    "efficientnet-l2",
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = (
            1 - global_params.batch_norm_momentum
        )  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = (
            block_args.id_skip
        )  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = (
            self._block_args.input_filters * self._block_args.expand_ratio
        )  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(
                1,
                int(
                    self._block_args.input_filters * self._block_args.se_ratio
                ),
            )
            self._se_reduce = Conv2d(
                in_channels=oup,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels,
                out_channels=oup,
                kernel_size=1,
            )

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
            self.id_skip
            and self._block_args.stride == 1
            and input_filters == output_filters
        ):
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(
                    x, p=drop_connect_rate, training=self.training
                )
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:


        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(
            32, self._global_params
        )  # number of output channels
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(
                    block_args.num_repeat, self._global_params
                ),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(
                    block_args, self._global_params, image_size=image_size
                )
            )
            image_size = calculate_output_image_size(
                image_size, block_args.stride
            )
            if (
                block_args.num_repeat > 1
            ):  # modify block_args to keep same output size
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(
                        block_args, self._global_params, image_size=image_size
                    )
                )
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x

        return endpoints


    def extract_endpoints_checkpointed(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate

            def checkpointer(function):
                def custom_forward(*inputs):
                    inputs = function(inputs[0], drop_connect_rate=drop_connect_rate)
                    return inputs
                return custom_forward

            x = checkpoint.checkpoint(checkpointer(block), x)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        # x = self.extract_features(inputs)
        x = self.extract_features_checkpointed(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(
            model_name, override_params
        )
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        weights_path=None,
        advprop=False,
        in_channels=3,
        num_classes=1000,
        **override_params
    ):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(
            model_name, num_classes=num_classes, **override_params
        )
        load_pretrained_weights(
            model,
            model_name,
            weights_path=weights_path,
            load_fc=(num_classes == 1000),
            advprop=advprop,
        )
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError(
                "model_name should be one of: " + ", ".join(VALID_MODELS)
            )

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(
                image_size=self._global_params.image_size
            )
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, bias=False
            )


class TwoWayFeaturePyramid(nn.Module):
    def __init__(
        self,
        en_pretrained_init=True,
        en_name="efficientnet-b5",
        activation=nn.LeakyReLU,
    ):
        super(TwoWayFeaturePyramid, self).__init__()
        if en_name != "efficientnet-b5":
            raise NotImplementedError(
                "model name is not implemented yet: {}".format(en_name)
            )

        if en_pretrained_init:
            self.en = EfficientNet.from_pretrained(en_name)
        else:
            self.en = EfficientNet.from_name(en_name)

        # Bottom-up branch
        self.times4_reduction_bu = conv_1x1_bn(40, 256)
        self.times8_reduction_bu = conv_1x1_bn(64, 256)
        self.times16_reduction_bu = conv_1x1_bn(176, 256)
        self.times32_reduction_bu = conv_1x1_bn(2048, 256)

        # Top-down branch
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.times4_reduction_td = conv_1x1_bn(40, 256)
        self.times8_reduction_td = conv_1x1_bn(64, 256)
        self.times16_reduction_td = conv_1x1_bn(176, 256)
        self.times32_reduction_td = conv_1x1_bn(2048, 256)

        # Ps Separable Convolutions
        self.p32conv = DepthSeparableConv2d(256, 256)
        self.p16conv = DepthSeparableConv2d(256, 256)
        self.p8conv = DepthSeparableConv2d(256, 256)
        self.p4conv = DepthSeparableConv2d(256, 256)

    def forward(self, inp):
        ini_shape = inp.shape
        end_points = self.en.extract_endpoints(inp)
        red_by_2 = end_points["reduction_1"]
        red_by_4 = end_points["reduction_2"]
        red_by_8 = end_points["reduction_3"]
        red_by_16 = end_points["reduction_4"]
        red_by_32 = end_points["reduction_5"]

        x_bu_1 = self.times4_reduction_bu(red_by_4)
        x_td_4_ = self.times4_reduction_td(red_by_4)

        x_bu_2 = self.times8_reduction_bu(red_by_8) + nn.AdaptiveAvgPool2d(
            (ini_shape[2] // 8, ini_shape[3] // 8)
        )(x_bu_1)
        x_td_3_ = self.times8_reduction_td(red_by_8)

        x_bu_3 = self.times16_reduction_bu(red_by_16) + nn.AdaptiveAvgPool2d(
            (ini_shape[2] // 16, ini_shape[3] // 16)
        )(x_bu_2)
        x_td_2_ = self.times16_reduction_td(red_by_16)

        x_bu_4 = self.times32_reduction_bu(red_by_32) + nn.AdaptiveAvgPool2d(
            (ini_shape[2] // 32, ini_shape[3] // 32)
        )(x_bu_3)
        x_td_1 = self.times32_reduction_td(red_by_32)

        # Top-down branch computation
        x_td_2 = x_td_2_ + self.upsample(x_td_1)
        x_td_3 = x_td_3_ + self.upsample(x_td_2)
        x_td_4 = x_td_4_ + self.upsample(x_td_3)

        # Final Ps
        p32 = self.p32conv(x_td_1 + x_bu_4)
        p16 = self.p16conv(x_td_2 + x_bu_3)
        p8 = self.p8conv(x_td_3 + x_bu_2)
        p4 = self.p4conv(x_td_4 + x_bu_1)

        return p32, p16, p8, p4


if __name__ == "__main__":
    en = EfficientNet.from_pretrained('efficientnet-b5').cuda()
    end_points = en.extract_endpoints_checkpointed(torch.rand(1, 3, 1024, 2048).cuda())
    print("reduction_1", end_points["reduction_1"].shape)
    print("reduction_2", end_points["reduction_2"].shape)
    print("reduction_3", end_points["reduction_3"].shape)
    print("reduction_4", end_points["reduction_4"].shape)
    print("reduction_5", end_points["reduction_5"].shape)

    # fpn = TwoWayFeaturePyramid().cuda()
    # p32, p16, p8, p4 = fpn(torch.rand(3, 3, 256, 512).cuda())
    # print("p32", p32.shape)
    # print("p16", p16.shape)
    # print("p8", p8.shape)
    # print("p4", p4.shape)
