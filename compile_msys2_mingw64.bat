set SRCPATH=C:\Users\tommie_cheng\Desktop\code\mathfunction\vulkan_kompute_test\thirdparty\kompute\src
g++ -DFMT_HEADER_ONLY -std=c++11 -o hello hello.cpp %SRCPATH%\Algorithm.cpp %SRCPATH%\Manager.cpp %SRCPATH%\OpAlgoDispatch.cpp %SRCPATH%\OpMemoryBarrier.cpp %SRCPATH%\OpTensorCopy.cpp %SRCPATH%\OpTensorSyncDevice.cpp %SRCPATH%\OpTensorSyncLocal.cpp %SRCPATH%\Sequence.cpp %SRCPATH%\Tensor.cpp ^
-I"C:\Users\tommie_cheng\Desktop\code\mathfunction\vulkan_kompute_test\thirdparty\kompute\src\include" ^
-I"C:\Users\tommie_cheng\Desktop\code\mathfunction\vulkan_kompute_test\thirdparty\fmt\include" ^
-L"C:\Users\tommie_cheng\Desktop\env\compiler\msys2\mingw64\lib" ^
-lvulkan.dll