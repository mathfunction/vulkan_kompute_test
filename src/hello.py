"""=============================================================================
    install VulkanSDK
    pip install pyshader
    pip install kp
================================================================================"""





import pyshader as ps
import kp
import numpy as np
import logging
kp_logger = logging.getLogger("kp")
#kp_logger.setLevel(logging.DEBUG)
kp_logger.setLevel(logging.INFO) 

#==================================================================================#
# write shader 
@ps.python2shader
def compute_mult(index=("input", "GlobalInvocationId", ps.ivec3),
                            data1=("buffer", 0, ps.Array(ps.f32)),
                            data2=("buffer", 1, ps.Array(ps.f32)),
                            data3=("buffer", 2, ps.Array(ps.f32))):
    i = index.x
    data3[i] = data1[i] * data2[i]
#byte_compute_mult = compute_mult.to_spirv()




byte_compute_mult = b'\x03\x02#\x07\x00\x03\x01\x00\x00\x00\x00\x00&\x00\x00\x00\x00\x00\x00\x00\x11\x00\x02\x00\x01\x00\x00\x00\x0e\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x06\x00\x05\x00\x00\x00\x01\x00\x00\x00main\x00\x00\x00\x00\x08\x00\x00\x00\x10\x00\x06\x00\x01\x00\x00\x00\x11\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x05\x00\x04\x00\x01\x00\x00\x00main\x00\x00\x00\x00\x05\x00\x04\x00\x08\x00\x00\x00index\x00\x00\x00\x05\x00\x04\x00\r\x00\x00\x00data1\x00\x00\x00\x05\x00\x03\x00\x0e\x00\x00\x000\x00\x00\x00\x05\x00\x04\x00\x11\x00\x00\x00data2\x00\x00\x00\x05\x00\x04\x00\x14\x00\x00\x00data3\x00\x00\x00\x05\x00\x03\x00\x16\x00\x00\x00i\x00\x00\x00G\x00\x04\x00\x08\x00\x00\x00\x0b\x00\x00\x00\x1c\x00\x00\x00G\x00\x04\x00\n\x00\x00\x00\x06\x00\x00\x00\x04\x00\x00\x00H\x00\x05\x00\x0b\x00\x00\x00\x00\x00\x00\x00#\x00\x00\x00\x00\x00\x00\x00G\x00\x03\x00\x0b\x00\x00\x00\x03\x00\x00\x00G\x00\x04\x00\r\x00\x00\x00"\x00\x00\x00\x00\x00\x00\x00G\x00\x04\x00\r\x00\x00\x00!\x00\x00\x00\x00\x00\x00\x00H\x00\x05\x00\x0f\x00\x00\x00\x00\x00\x00\x00#\x00\x00\x00\x00\x00\x00\x00G\x00\x03\x00\x0f\x00\x00\x00\x03\x00\x00\x00G\x00\x04\x00\x11\x00\x00\x00"\x00\x00\x00\x00\x00\x00\x00G\x00\x04\x00\x11\x00\x00\x00!\x00\x00\x00\x01\x00\x00\x00H\x00\x05\x00\x12\x00\x00\x00\x00\x00\x00\x00#\x00\x00\x00\x00\x00\x00\x00G\x00\x03\x00\x12\x00\x00\x00\x03\x00\x00\x00G\x00\x04\x00\x14\x00\x00\x00"\x00\x00\x00\x00\x00\x00\x00G\x00\x04\x00\x14\x00\x00\x00!\x00\x00\x00\x02\x00\x00\x00\x13\x00\x02\x00\x02\x00\x00\x00!\x00\x03\x00\x03\x00\x00\x00\x02\x00\x00\x00\x15\x00\x04\x00\x05\x00\x00\x00 \x00\x00\x00\x01\x00\x00\x00\x17\x00\x04\x00\x06\x00\x00\x00\x05\x00\x00\x00\x03\x00\x00\x00 \x00\x04\x00\x07\x00\x00\x00\x01\x00\x00\x00\x06\x00\x00\x00;\x00\x04\x00\x07\x00\x00\x00\x08\x00\x00\x00\x01\x00\x00\x00\x16\x00\x03\x00\t\x00\x00\x00 \x00\x00\x00\x1d\x00\x03\x00\n\x00\x00\x00\t\x00\x00\x00\x1e\x00\x03\x00\x0b\x00\x00\x00\n\x00\x00\x00 \x00\x04\x00\x0c\x00\x00\x00\x02\x00\x00\x00\x0b\x00\x00\x00;\x00\x04\x00\x0c\x00\x00\x00\r\x00\x00\x00\x02\x00\x00\x00+\x00\x04\x00\x05\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x1e\x00\x03\x00\x0f\x00\x00\x00\n\x00\x00\x00 \x00\x04\x00\x10\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00;\x00\x04\x00\x10\x00\x00\x00\x11\x00\x00\x00\x02\x00\x00\x00\x1e\x00\x03\x00\x12\x00\x00\x00\n\x00\x00\x00 \x00\x04\x00\x13\x00\x00\x00\x02\x00\x00\x00\x12\x00\x00\x00;\x00\x04\x00\x13\x00\x00\x00\x14\x00\x00\x00\x02\x00\x00\x00 \x00\x04\x00\x15\x00\x00\x00\x07\x00\x00\x00\x05\x00\x00\x00 \x00\x04\x00\x17\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00 \x00\x04\x00\x1d\x00\x00\x00\x02\x00\x00\x00\t\x00\x00\x00 \x00\x04\x00 \x00\x00\x00\x02\x00\x00\x00\t\x00\x00\x00 \x00\x04\x00$\x00\x00\x00\x02\x00\x00\x00\t\x00\x00\x006\x00\x05\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\xf8\x00\x02\x00\x04\x00\x00\x00;\x00\x04\x00\x15\x00\x00\x00\x16\x00\x00\x00\x07\x00\x00\x00A\x00\x05\x00\x17\x00\x00\x00\x18\x00\x00\x00\x08\x00\x00\x00\x0e\x00\x00\x00=\x00\x04\x00\x05\x00\x00\x00\x19\x00\x00\x00\x18\x00\x00\x00>\x00\x03\x00\x16\x00\x00\x00\x19\x00\x00\x00=\x00\x04\x00\x05\x00\x00\x00\x1a\x00\x00\x00\x16\x00\x00\x00=\x00\x04\x00\x05\x00\x00\x00\x1b\x00\x00\x00\x16\x00\x00\x00A\x00\x06\x00\x1d\x00\x00\x00\x1e\x00\x00\x00\r\x00\x00\x00\x0e\x00\x00\x00\x1a\x00\x00\x00=\x00\x04\x00\t\x00\x00\x00\x1f\x00\x00\x00\x1e\x00\x00\x00A\x00\x06\x00 \x00\x00\x00!\x00\x00\x00\x11\x00\x00\x00\x0e\x00\x00\x00\x1b\x00\x00\x00=\x00\x04\x00\t\x00\x00\x00"\x00\x00\x00!\x00\x00\x00\x85\x00\x05\x00\t\x00\x00\x00\x1c\x00\x00\x00\x1f\x00\x00\x00"\x00\x00\x00=\x00\x04\x00\x05\x00\x00\x00#\x00\x00\x00\x16\x00\x00\x00A\x00\x06\x00$\x00\x00\x00%\x00\x00\x00\x14\x00\x00\x00\x0e\x00\x00\x00#\x00\x00\x00>\x00\x03\x00%\x00\x00\x00\x1c\x00\x00\x00\xfd\x00\x01\x008\x00\x01\x00'


def test_array_multiplication(gpuIdx,bytecode):
    print("=============================================")
    # 1. Create Kompute Manager (selects device 0 by default)
    mgr = kp.Manager(gpuIdx)

    # 2. Create Kompute Tensors to hold data
    tensor_in_a = mgr.tensor(np.array([2, 2, 2,4]))
    tensor_in_b = mgr.tensor(np.array([1, 2, 3,5]))
    tensor_out = mgr.tensor(np.array([0,0,0,6]))
    params = [tensor_in_a, tensor_in_b, tensor_out]
    #  bytes = compute_mult.to_spirv()
    (
        mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(mgr.algorithm(params,bytecode)))
        .record(kp.OpTensorSyncLocal([tensor_out]))
        .eval()
    )

    return tensor_out.data()

tensor_out = test_array_multiplication(0,byte_compute_mult)
print(tensor_out)

#========================================================================================
