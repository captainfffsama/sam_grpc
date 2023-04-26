# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: samrpc.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0csamrpc.proto\x12\x0bsam_service\"%\n\x05Image\x12\x0e\n\x06imdata\x18\x01 \x01(\x0c\x12\x0c\n\x04path\x18\x02 \x01(\t\"g\n\x0eInputInferArgs\x12*\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12\x12\n\ninput_size\x18\x02 \x03(\x05\x12\x15\n\roriginal_size\x18\x03 \x03(\x05\"s\n\x17InputInferArgsWithCache\x12+\n\x06result\x18\x01 \x01(\x0b\x32\x1b.sam_service.InputInferArgs\x12+\n\tcache_idx\x18\x02 \x01(\x0b\x32\x18.sam_service.ServerCache\"\xa4\x02\n\x11SAMPredictRequest\x12/\n\ninfer_args\x18\x01 \x01(\x0b\x32\x1b.sam_service.InputInferArgs\x12,\n\x0cpoint_coords\x18\x02 \x01(\x0b\x32\x16.sam_service.TensorInt\x12,\n\x0cpoint_labels\x18\x03 \x01(\x0b\x32\x16.sam_service.TensorInt\x12#\n\x03\x62ox\x18\x04 \x01(\x0b\x32\x16.sam_service.TensorInt\x12,\n\nmask_input\x18\x05 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12\x18\n\x10multimask_output\x18\x06 \x01(\x08\x12\x15\n\rreturn_logits\x18\x07 \x01(\x08\"\xb5\x02\n\x19SAMPredictUseCacheRequest\x12\x32\n\x10infer_args_cache\x18\x01 \x01(\x0b\x32\x18.sam_service.ServerCache\x12,\n\x0cpoint_coords\x18\x02 \x01(\x0b\x32\x16.sam_service.TensorInt\x12,\n\x0cpoint_labels\x18\x03 \x01(\x0b\x32\x16.sam_service.TensorInt\x12#\n\x03\x62ox\x18\x04 \x01(\x0b\x32\x16.sam_service.TensorInt\x12\x32\n\x10mask_input_cache\x18\x05 \x01(\x0b\x32\x18.sam_service.ServerCache\x12\x18\n\x10multimask_output\x18\x06 \x01(\x08\x12\x15\n\rreturn_logits\x18\x07 \x01(\x08\"\xa0\x01\n\x12SAMPredictResponse\x12&\n\x05masks\x18\x01 \x01(\x0b\x32\x17.sam_service.TensorBool\x12(\n\x06scores\x18\x02 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12(\n\x06logits\x18\x03 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12\x0e\n\x06status\x18\x04 \x01(\x05\"{\n\x1bSAMPredictResponseWithCache\x12/\n\x06result\x18\x01 \x01(\x0b\x32\x1f.sam_service.SAMPredictResponse\x12+\n\tcache_idx\x18\x02 \x01(\x0b\x32\x18.sam_service.ServerCache\"5\n\x0bServerCache\x12\x12\n\ncache_name\x18\x01 \x01(\t\x12\x12\n\ncache_type\x18\x02 \x01(\t\"#\n\x11\x43leanCacheRespose\x12\x0e\n\x06status\x18\x01 \x01(\x05\"*\n\x0bTensorFloat\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x02\x12\r\n\x05shape\x18\x02 \x03(\x05\")\n\nTensorBool\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x08\x12\r\n\x05shape\x18\x02 \x03(\x05\"(\n\tTensorInt\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x05\x12\r\n\x05shape\x18\x02 \x03(\x05\x32\xb8\x03\n\nSAMService\x12I\n\x14SAMGetImageEmbedding\x12\x12.sam_service.Image\x1a\x1b.sam_service.InputInferArgs\"\x00\x12Z\n\x1cSAMGetImageEmbeddingUseCache\x12\x12.sam_service.Image\x1a$.sam_service.InputInferArgsWithCache\"\x00\x12O\n\nSAMPredict\x12\x1e.sam_service.SAMPredictRequest\x1a\x1f.sam_service.SAMPredictResponse\"\x00\x12h\n\x12SAMPredictUseCache\x12&.sam_service.SAMPredictUseCacheRequest\x1a(.sam_service.SAMPredictResponseWithCache\"\x00\x12H\n\nCleanCache\x12\x18.sam_service.ServerCache\x1a\x1e.sam_service.CleanCacheRespose\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'samrpc_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _IMAGE._serialized_start=29
  _IMAGE._serialized_end=66
  _INPUTINFERARGS._serialized_start=68
  _INPUTINFERARGS._serialized_end=171
  _INPUTINFERARGSWITHCACHE._serialized_start=173
  _INPUTINFERARGSWITHCACHE._serialized_end=288
  _SAMPREDICTREQUEST._serialized_start=291
  _SAMPREDICTREQUEST._serialized_end=583
  _SAMPREDICTUSECACHEREQUEST._serialized_start=586
  _SAMPREDICTUSECACHEREQUEST._serialized_end=895
  _SAMPREDICTRESPONSE._serialized_start=898
  _SAMPREDICTRESPONSE._serialized_end=1058
  _SAMPREDICTRESPONSEWITHCACHE._serialized_start=1060
  _SAMPREDICTRESPONSEWITHCACHE._serialized_end=1183
  _SERVERCACHE._serialized_start=1185
  _SERVERCACHE._serialized_end=1238
  _CLEANCACHERESPOSE._serialized_start=1240
  _CLEANCACHERESPOSE._serialized_end=1275
  _TENSORFLOAT._serialized_start=1277
  _TENSORFLOAT._serialized_end=1319
  _TENSORBOOL._serialized_start=1321
  _TENSORBOOL._serialized_end=1362
  _TENSORINT._serialized_start=1364
  _TENSORINT._serialized_end=1404
  _SAMSERVICE._serialized_start=1407
  _SAMSERVICE._serialized_end=1847
# @@protoc_insertion_point(module_scope)