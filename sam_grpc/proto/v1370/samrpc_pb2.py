# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: samrpc.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='samrpc.proto',
  package='sam_service',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0csamrpc.proto\x12\x0bsam_service\"%\n\x05Image\x12\x0e\n\x06imdata\x18\x01 \x01(\x0c\x12\x0c\n\x04path\x18\x02 \x01(\t\"g\n\x0eInputInferArgs\x12*\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12\x12\n\ninput_size\x18\x02 \x03(\x05\x12\x15\n\roriginal_size\x18\x03 \x03(\x05\"s\n\x17InputInferArgsWithCache\x12+\n\x06result\x18\x01 \x01(\x0b\x32\x1b.sam_service.InputInferArgs\x12+\n\tcache_idx\x18\x02 \x01(\x0b\x32\x18.sam_service.ServerCache\"\xa4\x02\n\x11SAMPredictRequest\x12/\n\ninfer_args\x18\x01 \x01(\x0b\x32\x1b.sam_service.InputInferArgs\x12,\n\x0cpoint_coords\x18\x02 \x01(\x0b\x32\x16.sam_service.TensorInt\x12,\n\x0cpoint_labels\x18\x03 \x01(\x0b\x32\x16.sam_service.TensorInt\x12#\n\x03\x62ox\x18\x04 \x01(\x0b\x32\x16.sam_service.TensorInt\x12,\n\nmask_input\x18\x05 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12\x18\n\x10multimask_output\x18\x06 \x01(\x08\x12\x15\n\rreturn_logits\x18\x07 \x01(\x08\"\xb5\x02\n\x19SAMPredictUseCacheRequest\x12\x32\n\x10infer_args_cache\x18\x01 \x01(\x0b\x32\x18.sam_service.ServerCache\x12,\n\x0cpoint_coords\x18\x02 \x01(\x0b\x32\x16.sam_service.TensorInt\x12,\n\x0cpoint_labels\x18\x03 \x01(\x0b\x32\x16.sam_service.TensorInt\x12#\n\x03\x62ox\x18\x04 \x01(\x0b\x32\x16.sam_service.TensorInt\x12\x32\n\x10mask_input_cache\x18\x05 \x01(\x0b\x32\x18.sam_service.ServerCache\x12\x18\n\x10multimask_output\x18\x06 \x01(\x08\x12\x15\n\rreturn_logits\x18\x07 \x01(\x08\"\xa0\x01\n\x12SAMPredictResponse\x12&\n\x05masks\x18\x01 \x01(\x0b\x32\x17.sam_service.TensorBool\x12(\n\x06scores\x18\x02 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12(\n\x06logits\x18\x03 \x01(\x0b\x32\x18.sam_service.TensorFloat\x12\x0e\n\x06status\x18\x04 \x01(\x05\"{\n\x1bSAMPredictResponseWithCache\x12/\n\x06result\x18\x01 \x01(\x0b\x32\x1f.sam_service.SAMPredictResponse\x12+\n\tcache_idx\x18\x02 \x01(\x0b\x32\x18.sam_service.ServerCache\"5\n\x0bServerCache\x12\x12\n\ncache_name\x18\x01 \x01(\t\x12\x12\n\ncache_type\x18\x02 \x01(\t\"#\n\x11\x43leanCacheRespose\x12\x0e\n\x06status\x18\x01 \x01(\x05\"*\n\x0bTensorFloat\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x02\x12\r\n\x05shape\x18\x02 \x03(\x05\")\n\nTensorBool\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x08\x12\r\n\x05shape\x18\x02 \x03(\x05\"(\n\tTensorInt\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x05\x12\r\n\x05shape\x18\x02 \x03(\x05\x32\xb8\x03\n\nSAMService\x12I\n\x14SAMGetImageEmbedding\x12\x12.sam_service.Image\x1a\x1b.sam_service.InputInferArgs\"\x00\x12Z\n\x1cSAMGetImageEmbeddingUseCache\x12\x12.sam_service.Image\x1a$.sam_service.InputInferArgsWithCache\"\x00\x12O\n\nSAMPredict\x12\x1e.sam_service.SAMPredictRequest\x1a\x1f.sam_service.SAMPredictResponse\"\x00\x12h\n\x12SAMPredictUseCache\x12&.sam_service.SAMPredictUseCacheRequest\x1a(.sam_service.SAMPredictResponseWithCache\"\x00\x12H\n\nCleanCache\x12\x18.sam_service.ServerCache\x1a\x1e.sam_service.CleanCacheRespose\"\x00\x62\x06proto3'
)




_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='sam_service.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='imdata', full_name='sam_service.Image.imdata', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='path', full_name='sam_service.Image.path', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=66,
)


_INPUTINFERARGS = _descriptor.Descriptor(
  name='InputInferArgs',
  full_name='sam_service.InputInferArgs',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='sam_service.InputInferArgs.features', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='input_size', full_name='sam_service.InputInferArgs.input_size', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='original_size', full_name='sam_service.InputInferArgs.original_size', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=68,
  serialized_end=171,
)


_INPUTINFERARGSWITHCACHE = _descriptor.Descriptor(
  name='InputInferArgsWithCache',
  full_name='sam_service.InputInferArgsWithCache',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='sam_service.InputInferArgsWithCache.result', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cache_idx', full_name='sam_service.InputInferArgsWithCache.cache_idx', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=173,
  serialized_end=288,
)


_SAMPREDICTREQUEST = _descriptor.Descriptor(
  name='SAMPredictRequest',
  full_name='sam_service.SAMPredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='infer_args', full_name='sam_service.SAMPredictRequest.infer_args', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='point_coords', full_name='sam_service.SAMPredictRequest.point_coords', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='point_labels', full_name='sam_service.SAMPredictRequest.point_labels', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='box', full_name='sam_service.SAMPredictRequest.box', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_input', full_name='sam_service.SAMPredictRequest.mask_input', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multimask_output', full_name='sam_service.SAMPredictRequest.multimask_output', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='return_logits', full_name='sam_service.SAMPredictRequest.return_logits', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=291,
  serialized_end=583,
)


_SAMPREDICTUSECACHEREQUEST = _descriptor.Descriptor(
  name='SAMPredictUseCacheRequest',
  full_name='sam_service.SAMPredictUseCacheRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='infer_args_cache', full_name='sam_service.SAMPredictUseCacheRequest.infer_args_cache', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='point_coords', full_name='sam_service.SAMPredictUseCacheRequest.point_coords', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='point_labels', full_name='sam_service.SAMPredictUseCacheRequest.point_labels', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='box', full_name='sam_service.SAMPredictUseCacheRequest.box', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_input_cache', full_name='sam_service.SAMPredictUseCacheRequest.mask_input_cache', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multimask_output', full_name='sam_service.SAMPredictUseCacheRequest.multimask_output', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='return_logits', full_name='sam_service.SAMPredictUseCacheRequest.return_logits', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=586,
  serialized_end=895,
)


_SAMPREDICTRESPONSE = _descriptor.Descriptor(
  name='SAMPredictResponse',
  full_name='sam_service.SAMPredictResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='masks', full_name='sam_service.SAMPredictResponse.masks', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scores', full_name='sam_service.SAMPredictResponse.scores', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='logits', full_name='sam_service.SAMPredictResponse.logits', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status', full_name='sam_service.SAMPredictResponse.status', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=898,
  serialized_end=1058,
)


_SAMPREDICTRESPONSEWITHCACHE = _descriptor.Descriptor(
  name='SAMPredictResponseWithCache',
  full_name='sam_service.SAMPredictResponseWithCache',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='sam_service.SAMPredictResponseWithCache.result', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cache_idx', full_name='sam_service.SAMPredictResponseWithCache.cache_idx', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1060,
  serialized_end=1183,
)


_SERVERCACHE = _descriptor.Descriptor(
  name='ServerCache',
  full_name='sam_service.ServerCache',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='cache_name', full_name='sam_service.ServerCache.cache_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cache_type', full_name='sam_service.ServerCache.cache_type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1185,
  serialized_end=1238,
)


_CLEANCACHERESPOSE = _descriptor.Descriptor(
  name='CleanCacheRespose',
  full_name='sam_service.CleanCacheRespose',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='sam_service.CleanCacheRespose.status', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1240,
  serialized_end=1275,
)


_TENSORFLOAT = _descriptor.Descriptor(
  name='TensorFloat',
  full_name='sam_service.TensorFloat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='sam_service.TensorFloat.data', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shape', full_name='sam_service.TensorFloat.shape', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1277,
  serialized_end=1319,
)


_TENSORBOOL = _descriptor.Descriptor(
  name='TensorBool',
  full_name='sam_service.TensorBool',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='sam_service.TensorBool.data', index=0,
      number=1, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shape', full_name='sam_service.TensorBool.shape', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1321,
  serialized_end=1362,
)


_TENSORINT = _descriptor.Descriptor(
  name='TensorInt',
  full_name='sam_service.TensorInt',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='sam_service.TensorInt.data', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shape', full_name='sam_service.TensorInt.shape', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1364,
  serialized_end=1404,
)

_INPUTINFERARGS.fields_by_name['features'].message_type = _TENSORFLOAT
_INPUTINFERARGSWITHCACHE.fields_by_name['result'].message_type = _INPUTINFERARGS
_INPUTINFERARGSWITHCACHE.fields_by_name['cache_idx'].message_type = _SERVERCACHE
_SAMPREDICTREQUEST.fields_by_name['infer_args'].message_type = _INPUTINFERARGS
_SAMPREDICTREQUEST.fields_by_name['point_coords'].message_type = _TENSORINT
_SAMPREDICTREQUEST.fields_by_name['point_labels'].message_type = _TENSORINT
_SAMPREDICTREQUEST.fields_by_name['box'].message_type = _TENSORINT
_SAMPREDICTREQUEST.fields_by_name['mask_input'].message_type = _TENSORFLOAT
_SAMPREDICTUSECACHEREQUEST.fields_by_name['infer_args_cache'].message_type = _SERVERCACHE
_SAMPREDICTUSECACHEREQUEST.fields_by_name['point_coords'].message_type = _TENSORINT
_SAMPREDICTUSECACHEREQUEST.fields_by_name['point_labels'].message_type = _TENSORINT
_SAMPREDICTUSECACHEREQUEST.fields_by_name['box'].message_type = _TENSORINT
_SAMPREDICTUSECACHEREQUEST.fields_by_name['mask_input_cache'].message_type = _SERVERCACHE
_SAMPREDICTRESPONSE.fields_by_name['masks'].message_type = _TENSORBOOL
_SAMPREDICTRESPONSE.fields_by_name['scores'].message_type = _TENSORFLOAT
_SAMPREDICTRESPONSE.fields_by_name['logits'].message_type = _TENSORFLOAT
_SAMPREDICTRESPONSEWITHCACHE.fields_by_name['result'].message_type = _SAMPREDICTRESPONSE
_SAMPREDICTRESPONSEWITHCACHE.fields_by_name['cache_idx'].message_type = _SERVERCACHE
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['InputInferArgs'] = _INPUTINFERARGS
DESCRIPTOR.message_types_by_name['InputInferArgsWithCache'] = _INPUTINFERARGSWITHCACHE
DESCRIPTOR.message_types_by_name['SAMPredictRequest'] = _SAMPREDICTREQUEST
DESCRIPTOR.message_types_by_name['SAMPredictUseCacheRequest'] = _SAMPREDICTUSECACHEREQUEST
DESCRIPTOR.message_types_by_name['SAMPredictResponse'] = _SAMPREDICTRESPONSE
DESCRIPTOR.message_types_by_name['SAMPredictResponseWithCache'] = _SAMPREDICTRESPONSEWITHCACHE
DESCRIPTOR.message_types_by_name['ServerCache'] = _SERVERCACHE
DESCRIPTOR.message_types_by_name['CleanCacheRespose'] = _CLEANCACHERESPOSE
DESCRIPTOR.message_types_by_name['TensorFloat'] = _TENSORFLOAT
DESCRIPTOR.message_types_by_name['TensorBool'] = _TENSORBOOL
DESCRIPTOR.message_types_by_name['TensorInt'] = _TENSORINT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.Image)
  })
_sym_db.RegisterMessage(Image)

InputInferArgs = _reflection.GeneratedProtocolMessageType('InputInferArgs', (_message.Message,), {
  'DESCRIPTOR' : _INPUTINFERARGS,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.InputInferArgs)
  })
_sym_db.RegisterMessage(InputInferArgs)

InputInferArgsWithCache = _reflection.GeneratedProtocolMessageType('InputInferArgsWithCache', (_message.Message,), {
  'DESCRIPTOR' : _INPUTINFERARGSWITHCACHE,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.InputInferArgsWithCache)
  })
_sym_db.RegisterMessage(InputInferArgsWithCache)

SAMPredictRequest = _reflection.GeneratedProtocolMessageType('SAMPredictRequest', (_message.Message,), {
  'DESCRIPTOR' : _SAMPREDICTREQUEST,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.SAMPredictRequest)
  })
_sym_db.RegisterMessage(SAMPredictRequest)

SAMPredictUseCacheRequest = _reflection.GeneratedProtocolMessageType('SAMPredictUseCacheRequest', (_message.Message,), {
  'DESCRIPTOR' : _SAMPREDICTUSECACHEREQUEST,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.SAMPredictUseCacheRequest)
  })
_sym_db.RegisterMessage(SAMPredictUseCacheRequest)

SAMPredictResponse = _reflection.GeneratedProtocolMessageType('SAMPredictResponse', (_message.Message,), {
  'DESCRIPTOR' : _SAMPREDICTRESPONSE,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.SAMPredictResponse)
  })
_sym_db.RegisterMessage(SAMPredictResponse)

SAMPredictResponseWithCache = _reflection.GeneratedProtocolMessageType('SAMPredictResponseWithCache', (_message.Message,), {
  'DESCRIPTOR' : _SAMPREDICTRESPONSEWITHCACHE,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.SAMPredictResponseWithCache)
  })
_sym_db.RegisterMessage(SAMPredictResponseWithCache)

ServerCache = _reflection.GeneratedProtocolMessageType('ServerCache', (_message.Message,), {
  'DESCRIPTOR' : _SERVERCACHE,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.ServerCache)
  })
_sym_db.RegisterMessage(ServerCache)

CleanCacheRespose = _reflection.GeneratedProtocolMessageType('CleanCacheRespose', (_message.Message,), {
  'DESCRIPTOR' : _CLEANCACHERESPOSE,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.CleanCacheRespose)
  })
_sym_db.RegisterMessage(CleanCacheRespose)

TensorFloat = _reflection.GeneratedProtocolMessageType('TensorFloat', (_message.Message,), {
  'DESCRIPTOR' : _TENSORFLOAT,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.TensorFloat)
  })
_sym_db.RegisterMessage(TensorFloat)

TensorBool = _reflection.GeneratedProtocolMessageType('TensorBool', (_message.Message,), {
  'DESCRIPTOR' : _TENSORBOOL,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.TensorBool)
  })
_sym_db.RegisterMessage(TensorBool)

TensorInt = _reflection.GeneratedProtocolMessageType('TensorInt', (_message.Message,), {
  'DESCRIPTOR' : _TENSORINT,
  '__module__' : 'samrpc_pb2'
  # @@protoc_insertion_point(class_scope:sam_service.TensorInt)
  })
_sym_db.RegisterMessage(TensorInt)



_SAMSERVICE = _descriptor.ServiceDescriptor(
  name='SAMService',
  full_name='sam_service.SAMService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1407,
  serialized_end=1847,
  methods=[
  _descriptor.MethodDescriptor(
    name='SAMGetImageEmbedding',
    full_name='sam_service.SAMService.SAMGetImageEmbedding',
    index=0,
    containing_service=None,
    input_type=_IMAGE,
    output_type=_INPUTINFERARGS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SAMGetImageEmbeddingUseCache',
    full_name='sam_service.SAMService.SAMGetImageEmbeddingUseCache',
    index=1,
    containing_service=None,
    input_type=_IMAGE,
    output_type=_INPUTINFERARGSWITHCACHE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SAMPredict',
    full_name='sam_service.SAMService.SAMPredict',
    index=2,
    containing_service=None,
    input_type=_SAMPREDICTREQUEST,
    output_type=_SAMPREDICTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SAMPredictUseCache',
    full_name='sam_service.SAMService.SAMPredictUseCache',
    index=3,
    containing_service=None,
    input_type=_SAMPREDICTUSECACHEREQUEST,
    output_type=_SAMPREDICTRESPONSEWITHCACHE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='CleanCache',
    full_name='sam_service.SAMService.CleanCache',
    index=4,
    containing_service=None,
    input_type=_SERVERCACHE,
    output_type=_CLEANCACHERESPOSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SAMSERVICE)

DESCRIPTOR.services_by_name['SAMService'] = _SAMSERVICE

# @@protoc_insertion_point(module_scope)
