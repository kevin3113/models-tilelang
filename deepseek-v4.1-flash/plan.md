## SuperKernel For TileLang

- 原理：算子二进制融合，基于AclGraph将多个算子合并为一个Task（函数调用方式）减少调度开销，同时进行亲和性优化提升性能，融合算子越多收益越大。  
- 芯片支持：910B/910C/950。  
- 语言支持：AscendC/PyPTO-PRO（适配中）、CANNBot-DSL（适配中）、TileLang（本地patch验证通过，codegen阶段相对独立的适配）。  
- 收益：950上平均每算子约1us优化，tilelang本地验证效果：AttenOut-FFNPre一段7个Tilelang算子*4层不同batch的收益3-11us，FFN段融合Tilelang的SwiGLU后提升3-6us。  
- 总结：TileLang支持SuperKernel后，首先可以带来decode阶段模型的执行收益，另外可以使得SuperKernel的优化保持最大化的收益。  
- 详细设计：[SuperKernel For Tilelang 补丁与验证](super-kernel-patch/SuperKernel-aclgraph-v2.md)  

## Tilelang 支持通信编程

- 目标：在 TileLang 中建立可扩展的通信编程语义，提供通信算子、通算融合算子开发，使同一份程序能够表达高阶集合通信、点对点、点到多点的通信能力，支持多后端。  
- 基本流程：通信域创建保留在torch框架，TileLang为通信类算子提供资源创建和持久化、通信发起、完成确认等能力。  
- 当前进展：基本通信模型的定义、通信前端接口原语扩展设计。  
- 详细设计：[TileLang通信编程设计思路](comm-ir-v2/TileLang-Comm-v2.md)  

## 其他优化点  

- 维测能力增强如aicore error自动dump输入tensor和shape信息（exception dump）。  
- 支持确定性计算以及batch一致性。  
- aicpu编程支持FA等seqlen更新的meta计算。  

## 当前几个重点PR  

- [支持batchmode 3614](https://github.com/tile-ai/tilelang/pull/3614)  
- [PTO后端切换ffi 3516](https://github.com/tile-ai/tilelang/pull/3516)  
- AIV拷出Schedule优化 [3616](https://github.com/tile-ai/tilelang/issues/3616) [3668](https://github.com/tile-ai/tilelang/pull/3668)  