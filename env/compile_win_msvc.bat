::set KOMPUTE_INCLUDE_DIR=C:\Users\tommie_cheng\Desktop\code\chungyo\tommie_2022\vulkan_kompute_test\thirdparty\kompute\single_include
echo compiling ...
set VULKAN_SDK_DIR=C:\Users\tommie_cheng\Desktop\code\thirdparty\VulkanSDK\1.2.198.1
set KOMPUTE_SRC=C:\Users\tommie_cheng\Desktop\code\chungyo\tommie_2022\vulkan_kompute_test\thirdparty\kompute\src
set FMT_INCLUDE=C:\Users\tommie_cheng\Desktop\code\chungyo\tommie_2022\vulkan_kompute_test\thirdparty\fmt-8.1.0\include
cl /std:c++20 /Fo"../bin/" /EHsc /O2 /c "../src/hello.cpp" "%KOMPUTE_SRC%\Algorithm.cpp" "%KOMPUTE_SRC%\Manager.cpp" "%KOMPUTE_SRC%\OpAlgoDispatch.cpp" "%KOMPUTE_SRC%\OpMemoryBarrier.cpp" "%KOMPUTE_SRC%\OpTensorCopy.cpp" "%KOMPUTE_SRC%\OpTensorSyncDevice.cpp" "%KOMPUTE_SRC%\OpTensorSyncLocal.cpp" "%KOMPUTE_SRC%\Sequence.cpp" "%KOMPUTE_SRC%\Tensor.cpp" ^
/D FMT_HEADER_ONLY ^
/I "%VULKAN_SDK_DIR%\Include" ^
/I "%KOMPUTE_SRC%\include" ^
/I "%FMT_INCLUDE%"
echo linking ...
link /libpath:"%VULKAN_SDK_DIR%\Lib" vulkan-1.lib ^
../bin/Algorithm.obj ^
../bin/Manager.obj ^
../bin/OpAlgoDispatch.obj ^
../bin/OpMemoryBarrier.obj ^
../bin/OpTensorSyncDevice.obj ^
../bin/OpTensorSyncLocal.obj ^
../bin/Sequence.obj ^
../bin/hello.obj ^
/out:"../bin/hello_win_msvc.exe"