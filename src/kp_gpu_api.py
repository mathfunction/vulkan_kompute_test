"""============================================================================================


============================================================================================="""
import os
import logging
import numpy as np
import kp # vulkan-kompute (vulkanSDK , cmake in terminal)
def compileShader(code):
	open("tmp_kp_shader.comp", "w").write(code)
	os.system("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
	# 利用 glslangValidator 編譯 // 安裝在 vulkan SDK 裡面 !!
	spirv_bytes = open("tmp_kp_shader.comp.spv", "rb").read()
	os.remove("tmp_kp_shader.comp")
	os.remove("tmp_kp_shader.comp.spv")
	return spirv_bytes

#---------------------------------------------------------------------------------------------
class NxNMatMulShader:
	def __init__(self,n,gpuIdx=0,openLog=False):
		if openLog == True:
			self.kp_logger = logging.getLogger("kp")
			self.kp_logger.setLevel(logging.INFO)
		matmul_shader_code = """
				#version 450
				layout (local_size_x = 1, local_size_y = 1) in;
				layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };
				layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float in_tensor_2[]; };
				layout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float out_tensor[]; };
				layout (constant_id = 0) const float tensor_size_f = 0;
				void main()
				{
					uint globalRow = gl_GlobalInvocationID.x;
					uint globalCol = gl_GlobalInvocationID.y;
					uint tensor_size = uint(tensor_size_f);
					float acc = 0.0;
					for(uint k = 0u; k < tensor_size; k++)
						acc += in_tensor_1[(k * tensor_size) + globalRow] * in_tensor_2[(globalCol * tensor_size) + k];
					out_tensor[(globalCol * tensor_size) + globalRow] = acc;
				}
		"""
		self.mgr = kp.Manager(gpuIdx)
		self.matmul_shader_bytes = compileShader(matmul_shader_code)
		self.tensor_size = n
		self.tensor_shape = [n,n]
		self.C = self.mgr.tensor(np.zeros(self.tensor_shape))
	# A,B are nxn	
	def matmul(self,A,B):
		paras = [self.mgr.tensor(B),self.mgr.tensor(A),self.C] 
		# define algo
		algo = self.mgr.algorithm(
			paras,  # params
			self.matmul_shader_bytes,  # spirv
			(*self.tensor_shape, 1),  # workgroup
			[float(self.tensor_size)],  # spec_consts
			[]
		)  # push_consts
		# computing ...
		(
			self.mgr.sequence()
		 	.record(kp.OpTensorSyncDevice(paras))
		 	.record(kp.OpAlgoDispatch(algo))
		 	.record(kp.OpTensorSyncLocal(paras))
		 	.eval()
		)
		return self.C.data().reshape(self.tensor_shape)
#------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	import timeit
	n = 4
	shader = NxNMatMulShader(n,openLog=False)
	for i in range(5):
		print("================================================")
		print(f"[{i}]")
		print("================================================") 
		A = np.array(np.random.randn(n,n))
		B = np.array(np.random.randn(n,n))
		t1 = timeit.default_timer()
		C1 = A@B
		t2 = timeit.default_timer()
		C2 = shader.matmul(A,B)
		t3 = timeit.default_timer()
		

		print(f"pure_numpy:{C1},{(t2-t1)*1000} ms")
		print(f"kp_shader:{C2},{(t3-t2)*1000}ms")

		








































if __name__ == '__main__':
	pass
