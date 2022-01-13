"""============================================================================================
compute shader 
https://www.khronos.org/opengl/wiki/Compute_Shader#Overview
https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm.htm
https://cnugteren.github.io/tutorial/pages/page1.html
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
	# 刪掉 
	os.remove("tmp_kp_shader.comp")
	os.remove("tmp_kp_shader.comp.spv")
	return spirv_bytes

#---------------------------------------------------------------------------------------------
class SGEMMShader:
	def __init__(self,MKN,gpuIdx=0,logLevel=logging.NOTSET):
		self.kp_logger = logging.getLogger("kp")
		self.kp_logger.setLevel(logLevel)
		naive_matmul_shader_code = """
				#version 450
				layout (local_size_x = 1, local_size_y = 1) in;
				layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float A[]; };
				layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float B[]; };
				layout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float C[]; };
				layout (constant_id = 0) const float Mf = 0;
				layout (constant_id = 1) const float Kf = 0;
				layout (constant_id = 2) const float Nf = 0;
				void main(){
					uint m = gl_GlobalInvocationID.x;
					uint n = gl_GlobalInvocationID.y;
					uint M = uint(Mf);
					uint K = uint(Kf);
					uint N = uint(Nf);
					float AB_mn = 0.0f;
					for(uint k = 0u; k < K; k++)
						AB_mn += A[k*M+m]*B[n*K+k]; // parallel with (M,N,1)
					C[(m*N)+n] = AB_mn;
				}//end_main
		"""
		TS = 32
		WPT = 8
		RTS = 4
		opt_shader_code = '''
		#version 450
		layout (local_size_x = {TS}, local_size_y = {RTS}) in;
		layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
		layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
		layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
		layout (constant_id = 0) const float Mf = 0;
		layout (constant_id = 1) const float Kf = 0;
		layout (constant_id = 2) const float Nf = 0;

		shared float sub_tensor_1[{TS}][{TS}];
		shared float sub_tensor_2[{TS}][{TS}];
		void main(){{
		    uint row = gl_LocalInvocationID.x; // 0 .. TS
		    uint col = gl_LocalInvocationID.y; // 0 .. TS
		    // gl_WorkGroupID : 0 .. tensor_size // TS
		    uint globalRow = {TS} * gl_WorkGroupID.x + row;
		    uint globalCol = {TS} * gl_WorkGroupID.y + col;
		    uint M = uint(Mf);
		    uint K = uint(Kf);
		    uint N = uint(Nf);
		    
		    // Initialise the accumulation registers
    		float acc[{WPT}];
    		for (uint w=0u; w < {WPT}; w++) {{
        		acc[w] = 0.0;
    		}}//endfor
		    //float acc = 0.0;
		    uint numTiles = K / {TS};
		    for(uint t = 0u; t < numTiles; t++){{
		        for (uint w=0u; w < {WPT}; w++) {{
			        uint tiledRow = ({TS} * t) + row;
			        uint tiledCol = ({TS} * t) + col;
			        sub_tensor_1[col+w*{RTS}][row] = in_tensor_1[(tiledCol+w*{RTS})*M + globalRow];
			        sub_tensor_2[col+w*{RTS}][row] = in_tensor_2[(globalCol+w*{RTS})*K + tiledRow];
			    }}//endfor
		        memoryBarrierShared();
		        barrier();
		        for(uint k = 0u; k < {TS}; k++){{
		        	for (uint w=0u; w < {WPT}; w++) {{
		            	acc[w] += sub_tensor_1[k][row] * sub_tensor_2[col+w*{RTS}][k];
		        	}}
		        }}
		        barrier();
		    }}//endfor
		    // Store the final result in C
		    for(uint w=0u; w < {WPT}; w++) {{
		    	out_tensor[M*(globalCol+w*{RTS}) + globalRow] = acc[w];
		    }}//endfor
		}}'''
		self.mgr = kp.Manager(gpuIdx)
		self.matmul_shader_bytes = compileShader(opt_shader_code.format(TS=TS,WPT=WPT,RTS=RTS))
		#self.matmul_shader_bytes = compileShader(naive_matmul_shader_code)
		M,K,N = MKN
		self.returnShape = [M,N]
		# define tensors 
		self.kpA = self.mgr.tensor(np.zeros([K,M]))
		self.kpB = self.mgr.tensor(np.zeros([N,K]))
		self.kpC = self.mgr.tensor(np.zeros([M,N]))
		
		# define algorithm 
		algo = self.mgr.algorithm(
			[self.kpA,self.kpB,self.kpC],  # params
			self.matmul_shader_bytes,  # spirv
			(M//TS,N//TS,1),  # workgroup
			[float(M),float(K),float(N)],  # spec_consts
			[]
		)  # push_consts
		self.algoPipline = self.mgr.sequence()
		self.algoPipline.record(kp.OpTensorSyncDevice([self.kpA,self.kpB])) # map Tensor to GPU
		self.algoPipline.record(kp.OpAlgoDispatch(algo))
		self.algoPipline.record(kp.OpTensorSyncLocal([self.kpC])) # map GPU to Tensor

	def showDevice(self):
		print("#-----------------------------------#")
		d = self.mgr.get_device_properties()
		for k in d:
			print(f"[{k}]:{d[k]}")
		print("#-----------------------------------#")


	# A,B are nxn	
	def matmul(self,A,B):
		# copy data 
		self.kpA.data()[:] = A.reshape(-1)
		self.kpB.data()[:] = B.reshape(-1)
		self.algoPipline.eval()
		return self.kpC.data().reshape(self.returnShape)





if __name__ == '__main__':
	
	def GPU_Algorithm():
		"""=================================================================================================================================================
		issue : libc++abi: terminating with uncaught exception of type pybind11::error_already_set: TypeError: 'NoneType' object is not callable ,
		實測: 最外層需要有一個  def 函式包裝 !!
		=================================================================================================================================================="""
		import timeit
		import traceback
		try:
			M = 1024
			K = 1024
			N = 1024
			shader = SGEMMShader([M,K,N],gpuIdx=0,logLevel=logging.INFO)
			shader.showDevice()
			for i in range(3):
				print("================================================")
				print(f"[{i}]")
				print("================================================") 
				A = np.array(np.random.randn(K,M)).astype(float)
				B = np.array(np.random.randn(N,K)).astype(float)
				t1 = timeit.default_timer()
				C1 = B@A
				t2 = timeit.default_timer()
				C2 = shader.matmul(A,B)
				t3 = timeit.default_timer()
				if i == 0:
					print(f"pure_cpu_numpy:{C1}")
					print(f"kp_gpu_shader:{C2}")
				else:
					print(f"pure_cpu_numpy:{(t2-t1)*1000} ms")
					print(f"kp_gpu_shader:{(t3-t2)*1000}ms")
		except Exception as e:
			print(e)
			print(traceback.format_exc())
	GPU_Algorithm()








































if __name__ == '__main__':
	pass
