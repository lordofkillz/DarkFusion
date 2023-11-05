#For converting models Install at your own risk if it fails pip install --user <package-name> put here then put C:\Users\jdini\AppData\Roaming\Python\Python38\Scripts
C:\Users\jdini\AppData\Roaming\Python\Python38\site-packages
in system environment variable path. 
# to convert to .engine or .tflite you must have pip install protobuf==3.20.3 installed

to instal tensorRT put pip install 
pip install python\tensorrt-8.6.1-cp38-none-win_amd64.whl
pip install graphsurgeon\graphsurgeon-0.4.6-py2.py3-none-any.whl
pip install uff\uff-0.6.9-py2.py3-none-any.whl
pip install onnx_graphsurgeon\onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
leave Tensort in workign direcotry and put \TensorRT\lib in system env path.

pip install these recommended as needed.
openvino==2023.1.0
openvino-dev==2023.1.0
openvino-telemetry==2023.1.1
onnxsim>=0.4.33
tensorflow==2.13.0 
onnx2tf>=1.15.4
sng4onnx>=1.0.1
onnx_graphsurgeon>=0.3.26
tflite_support
paddlepaddle
x2paddle
ncnn