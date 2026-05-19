from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'yolov8n.onnx'
model_quant = 'yolov8n_int8.onnx'
quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
print("Quantization to INT8 complete!")
