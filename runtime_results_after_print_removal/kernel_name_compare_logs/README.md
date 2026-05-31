# Kernel name comparison logs

Generated from Nsight Systems SQLite exports using:

```sql
select StringIds.value, count(*), sum(end-start), avg(end-start)
from CUPTI_ACTIVITY_KIND_KERNEL join StringIds on demangledName = StringIds.id
group by StringIds.value
order by StringIds.value;
```

## resnet

- fresh kernel-name log: `resnet_custom_conv_fresh_kernel_names.csv`
- cutlass_test kernel-name log: `resnet_custom_conv_cutlass_test_kernel_names.csv`
- name/count diff log: `resnet_fresh_vs_cutlass_test_kernel_name_count_diff.csv`
- diff rows excluding header: `0`

## mobilenet

- fresh kernel-name log: `mobilenet_custom_conv_fresh_kernel_names.csv`
- cutlass_test kernel-name log: `mobilenet_custom_conv_cutlass_test_kernel_names.csv`
- name/count diff log: `mobilenet_fresh_vs_cutlass_test_kernel_name_count_diff.csv`
- diff rows excluding header: `0`

## shufflenet

- fresh kernel-name log: `shufflenet_custom_conv_fresh_kernel_names.csv`
- cutlass_test kernel-name log: `shufflenet_custom_conv_cutlass_test_kernel_names.csv`
- name/count diff log: `shufflenet_fresh_vs_cutlass_test_kernel_name_count_diff.csv`
- diff rows excluding header: `0`
