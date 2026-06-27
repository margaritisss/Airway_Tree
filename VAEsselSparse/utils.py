class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'torchsparse' not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)

        spatial_range = out.spatial_range

        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
        out._spatial_cache = x._spatial_cache
        out._scale = tuple([s * stride for s, stride in zip(x._scale, self.conv.stride)])

        out.data.spatial_range = spatial_range

        return out




class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        algo = None
        if SPCONV_ALGO == 'native':
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == 'implicit_gemm':
            algo = spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1 and (padding is None):
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
        else:
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:

        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
        new_data = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
            sorted_feats = new_data.features #[fwd]
            sorted_coords = new_data.indices #[fwd]
            unsorted_data = new_data

            indice_dict = new_data.indice_dict 
            
            if 'spconv' not in globals():
                import spconv.pytorch as spconv
            new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size, indice_dict=indice_dict)  # type: ignore

        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
            out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)
 
        return out
