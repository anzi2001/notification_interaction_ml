Ϡ
�+�*
Y
AddN
inputs"T*N
sum"T"
Nint(0"#
Ttype:
2	��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype�
E
AssignSubVariableOp
resource
value"dtype"
dtypetype�
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
�
BiasAddGrad
out_backprop"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
.
Log1p
x"T
y"T"
Ttype:

2
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
7

Reciprocal
x"T
y"T"
Ttype:
2
	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.16.12v2.16.1-0-g5bc9d26649c8��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_4/bias/*
dtype0*
shape:*$
shared_nameAdam/v/dense_4/bias
w
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_4/bias/*
dtype0*
shape:*$
shared_nameAdam/m/dense_4/bias
w
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_4/kernel/*
dtype0*
shape
:@*&
shared_nameAdam/v/dense_4/kernel

)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_4/kernel/*
dtype0*
shape
:@*&
shared_nameAdam/m/dense_4/kernel

)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel*
_output_shapes

:@*
dtype0
�
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_3/bias/*
dtype0*
shape:@*$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_3/bias/*
dtype0*
shape:@*$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_3/kernel/*
dtype0*
shape:	�@*&
shared_nameAdam/v/dense_3/kernel
�
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_3/kernel/*
dtype0*
shape:	�@*&
shared_nameAdam/m/dense_3/kernel
�
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_2/bias/*
dtype0*
shape:�*$
shared_nameAdam/v/dense_2/bias
x
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_2/bias/*
dtype0*
shape:�*$
shared_nameAdam/m/dense_2/bias
x
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_2/kernel/*
dtype0*
shape:
��*&
shared_nameAdam/v/dense_2/kernel
�
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_2/kernel/*
dtype0*
shape:
��*&
shared_nameAdam/m/dense_2/kernel
�
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_1/bias/*
dtype0*
shape:�*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_1/bias/*
dtype0*
shape:�*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_1/kernel/*
dtype0*
shape:	�*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_1/kernel/*
dtype0*
shape:	�*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/v/dense/bias/*
dtype0*
shape:*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/m/dense/bias/*
dtype0*
shape:*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense/kernel/*
dtype0*
shape
:*$
shared_nameAdam/v/dense/kernel
{
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense/kernel/*
dtype0*
shape
:*$
shared_nameAdam/m/dense/kernel
{
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes

:*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
�
dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape
:@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@*
dtype0
�
dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
�
dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape:	�@*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	�@*
dtype0
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
j
infer_xPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallinfer_xdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_1884
�
StatefulPartitionedCall_1StatefulPartitionedCalldense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *j
_output_shapesX
V:::	�:�:
��:�:	�@:@:@:*,
_read_only_resource_inputs

 	*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_1927
[

restore_a0Placeholder*
_output_shapes

:*
dtype0*
shape
:
S

restore_a1Placeholder*
_output_shapes
:*
dtype0*
shape:
]

restore_a2Placeholder*
_output_shapes
:	�*
dtype0*
shape:	�
U

restore_a3Placeholder*
_output_shapes	
:�*
dtype0*
shape:�
_

restore_a4Placeholder* 
_output_shapes
:
��*
dtype0*
shape:
��
U

restore_a5Placeholder*
_output_shapes	
:�*
dtype0*
shape:�
]

restore_a6Placeholder*
_output_shapes
:	�@*
dtype0*
shape:	�@
S

restore_a7Placeholder*
_output_shapes
:@*
dtype0*
shape:@
[

restore_a8Placeholder*
_output_shapes

:@*
dtype0*
shape
:@
S

restore_a9Placeholder*
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCall_2StatefulPartitionedCall
restore_a0
restore_a1
restore_a2
restore_a3
restore_a4
restore_a5
restore_a6
restore_a7
restore_a8
restore_a9dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2
*
_collective_manager_ids
 *j
_output_shapesX
V:::	�:�:
��:�:	�@:@:@:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_1980
j
train_xPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
j
train_yPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_3StatefulPartitionedCalltrain_xtrain_ydense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biastotal_1count_1	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biastotalcount*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_1858

NoOpNoOp
�O
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�O
value�OB�O B�O
R
	model
	infer

parameters
restore
	train

signatures*
�
layer-0
layer_with_weights-0
layer-1
	layer_with_weights-1
	layer-2

layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer_with_weights-4
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer*

trace_0* 

trace_0* 

trace_0* 

trace_0* 
5
	train
	infer

parameters
restore* 
* 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator* 
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
E_random_generator* 
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator* 
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias*
J
&0
'1
.2
/3
=4
>5
L6
M7
[8
\9*
J
&0
'1
.2
/3
=4
>5
L6
M7
[8
\9*
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

btrace_0
ctrace_1* 

dtrace_0
etrace_1* 
* 
�
f
_variables
g_iterations
h_learning_rate
i_index_dict
j
_momentums
k_velocities
l_update_step_xla*
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*

&0
'1*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 
b\
VARIABLE_VALUEdense/kernel<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
dense/bias:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 
d^
VARIABLE_VALUEdense_1/kernel<model/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEdense_1/bias:model/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEdense_2/kernel<model/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEdense_2/bias:model/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

L0
M1*

L0
M1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEdense_3/kernel<model/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEdense_3/bias:model/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

[0
\1*

[0
\1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEdense_4/kernel<model/layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEdense_4/bias:model/layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
	2

3
4
5
6
7
8*

�0
�1*
* 
* 
* 
* 
* 
* 
�
g0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20*
YS
VARIABLE_VALUE	iteration6model/optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUElearning_rate9model/optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
d^
VARIABLE_VALUEAdam/m/dense/kernel7model/optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense/kernel7model/optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense/bias7model/optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense/bias7model/optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/dense_1/kernel7model/optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/dense_1/kernel7model/optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_1/bias7model/optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_1/bias7model/optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/dense_2/kernel7model/optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/dense_2/kernel8model/optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/dense_2/bias8model/optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/dense_2/bias8model/optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/dense_3/kernel8model/optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/dense_3/kernel8model/optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/dense_3/bias8model/optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/dense_3/bias8model/optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/dense_4/kernel8model/optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/dense_4/kernel8model/optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/dense_4/bias8model/optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/dense_4/bias8model/optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
[U
VARIABLE_VALUEtotal_1:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcount_1:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
YS
VARIABLE_VALUEtotal:model/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEcount:model/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biastotal_1count_1totalcountConst*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_3724
�
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biastotal_1count_1totalcount*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_3841ژ
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_3076

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�2
�
__inference_restore_2137
a0
a1
a2
a3
a4
a5
a6
a7
a8
a9+
assignvariableop_resource:)
assignvariableop_1_resource:.
assignvariableop_2_resource:	�*
assignvariableop_3_resource:	�/
assignvariableop_4_resource:
��*
assignvariableop_5_resource:	�.
assignvariableop_6_resource:	�@)
assignvariableop_7_resource:@-
assignvariableop_8_resource:@)
assignvariableop_9_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�StatefulPartitionedCall|
AssignVariableOpAssignVariableOpassignvariableop_resourcea0*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcea1*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_2AssignVariableOpassignvariableop_2_resourcea2*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_3AssignVariableOpassignvariableop_3_resourcea3*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_4AssignVariableOpassignvariableop_4_resourcea4*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_5AssignVariableOpassignvariableop_5_resourcea5*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_6AssignVariableOpassignvariableop_6_resourcea6*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_7AssignVariableOpassignvariableop_7_resourcea7*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_8AssignVariableOpassignvariableop_8_resourcea8*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_9AssignVariableOpassignvariableop_9_resourcea9*
_output_shapes
 *
dtype0*
validate_shape(�
StatefulPartitionedCallStatefulPartitionedCallassignvariableop_resourceassignvariableop_1_resourceassignvariableop_2_resourceassignvariableop_3_resourceassignvariableop_4_resourceassignvariableop_5_resourceassignvariableop_6_resourceassignvariableop_7_resourceassignvariableop_8_resourceassignvariableop_9_resource^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *j
_output_shapesX
V:::	�:�:
��:�:	�@:@:@:*,
_read_only_resource_inputs

 	*-
config_proto

CPU

GPU 2J 8� *$
fR
__inference_parameters_2085f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:i

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
:	�e

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*
_output_shapes	
:�j

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0* 
_output_shapes
:
��e

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*
_output_shapes	
:�i

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*
_output_shapes
:	�@d

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*
_output_shapes
:@h

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*
_output_shapes

:@d

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:::	�:�:
��:�:	�@:@:@:: : : : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:>	:

_output_shapes
:

_user_specified_namea9:B>

_output_shapes

:@

_user_specified_namea8:>:

_output_shapes
:@

_user_specified_namea7:C?

_output_shapes
:	�@

_user_specified_namea6:?;

_output_shapes	
:�

_user_specified_namea5:D@
 
_output_shapes
:
��

_user_specified_namea4:?;

_output_shapes	
:�

_user_specified_namea3:C?

_output_shapes
:	�

_user_specified_namea2:>:

_output_shapes
:

_user_specified_namea1:B >

_output_shapes

:

_user_specified_namea0
�
a
(__inference_dropout_1_layer_call_fn_3237

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_2923p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
?__inference_dense_layer_call_and_return_conditional_losses_2861

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
?__inference_model_layer_call_and_return_conditional_losses_2971
input_1

dense_2862:

dense_2864:
dense_1_2878:	�
dense_1_2880:	� 
dense_2_2907:
��
dense_2_2909:	�
dense_3_2936:	�@
dense_3_2938:@
dense_4_2965:@
dense_4_2967:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1
dense_2862
dense_2864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2861�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2878dense_1_2880*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2877�
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2894�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_2907dense_2_2909*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2906�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_2923�
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_2936dense_3_2938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2935�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2952�
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_2965dense_4_2967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2964w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:$
 

_user_specified_name2967:$	 

_user_specified_name2965:$ 

_user_specified_name2938:$ 

_user_specified_name2936:$ 

_user_specified_name2909:$ 

_user_specified_name2907:$ 

_user_specified_name2880:$ 

_user_specified_name2878:$ 

_user_specified_name2864:$ 

_user_specified_name2862:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
"__inference_signature_wrapper_1980
a0
a1
a2
a3
a4
a5
a6
a7
a8
a9
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalla0a1a2a3a4a5a6a7a8a9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2
*
_collective_manager_ids
 *j
_output_shapesX
V:::	�:�:
��:�:	�@:@:@:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� * 
fR
__inference_restore_298f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:i

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
:	�e

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*
_output_shapes	
:�j

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0* 
_output_shapes
:
��e

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*
_output_shapes	
:�i

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*
_output_shapes
:	�@d

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*
_output_shapes
:@h

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*
_output_shapes

:@d

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:::	�:�:
��:�:	�@:@:@:: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1958:$ 

_user_specified_name1956:$ 

_user_specified_name1954:$ 

_user_specified_name1952:$ 

_user_specified_name1950:$ 

_user_specified_name1948:$ 

_user_specified_name1946:$ 

_user_specified_name1944:$ 

_user_specified_name1942:$
 

_user_specified_name1940:>	:

_output_shapes
:

_user_specified_namea9:B>

_output_shapes

:@

_user_specified_namea8:>:

_output_shapes
:@

_user_specified_namea7:C?

_output_shapes
:	�@

_user_specified_namea6:?;

_output_shapes	
:�

_user_specified_namea5:D@
 
_output_shapes
:
��

_user_specified_namea4:?;

_output_shapes	
:�

_user_specified_namea3:C?

_output_shapes
:	�

_user_specified_namea2:>:

_output_shapes
:

_user_specified_namea1:B >

_output_shapes

:

_user_specified_namea0
�'
�
!__inference_internal_grad_fn_3561
result_grads_0
result_grads_1
result_grads_2
result_grads_3
result_grads_4
result_grads_5
result_grads_6
result_grads_7
result_grads_8
result_grads_9
result_grads_10
result_grads_11
result_grads_12
result_grads_13
result_grads_14
result_grads_15
result_grads_16
result_grads_17
result_grads_18
result_grads_19
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19M
IdentityIdentityresult_grads_0*
T0*
_output_shapes

:K

Identity_1Identityresult_grads_1*
T0*
_output_shapes
:P

Identity_2Identityresult_grads_2*
T0*
_output_shapes
:	�L

Identity_3Identityresult_grads_3*
T0*
_output_shapes	
:�Q

Identity_4Identityresult_grads_4*
T0* 
_output_shapes
:
��L

Identity_5Identityresult_grads_5*
T0*
_output_shapes	
:�P

Identity_6Identityresult_grads_6*
T0*
_output_shapes
:	�@K

Identity_7Identityresult_grads_7*
T0*
_output_shapes
:@O

Identity_8Identityresult_grads_8*
T0*
_output_shapes

:@K

Identity_9Identityresult_grads_9*
T0*
_output_shapes
:�
	IdentityN	IdentityNresult_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9*
T
2**
_gradient_op_typeCustomGradient-3520*�
_output_shapes�
�:::	�:�:
��:�:	�@:@:@::::	�:�:
��:�:	�@:@:@:T
Identity_10IdentityIdentityN:output:0*
T0*
_output_shapes

:P
Identity_11IdentityIdentityN:output:1*
T0*
_output_shapes
:U
Identity_12IdentityIdentityN:output:2*
T0*
_output_shapes
:	�Q
Identity_13IdentityIdentityN:output:3*
T0*
_output_shapes	
:�V
Identity_14IdentityIdentityN:output:4*
T0* 
_output_shapes
:
��Q
Identity_15IdentityIdentityN:output:5*
T0*
_output_shapes	
:�U
Identity_16IdentityIdentityN:output:6*
T0*
_output_shapes
:	�@P
Identity_17IdentityIdentityN:output:7*
T0*
_output_shapes
:@T
Identity_18IdentityIdentityN:output:8*
T0*
_output_shapes

:@P
Identity_19IdentityIdentityN:output:9*
T0*
_output_shapes
:"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:::	�:�:
��:�:	�@:@:@::::	�:�:
��:�:	�@:@:@::KG

_output_shapes
:
)
_user_specified_nameresult_grads_19:OK

_output_shapes

:@
)
_user_specified_nameresult_grads_18:KG

_output_shapes
:@
)
_user_specified_nameresult_grads_17:PL

_output_shapes
:	�@
)
_user_specified_nameresult_grads_16:LH

_output_shapes	
:�
)
_user_specified_nameresult_grads_15:QM
 
_output_shapes
:
��
)
_user_specified_nameresult_grads_14:LH

_output_shapes	
:�
)
_user_specified_nameresult_grads_13:PL

_output_shapes
:	�
)
_user_specified_nameresult_grads_12:KG

_output_shapes
:
)
_user_specified_nameresult_grads_11:O
K

_output_shapes

:
)
_user_specified_nameresult_grads_10:J	F

_output_shapes
:
(
_user_specified_nameresult_grads_9:NJ

_output_shapes

:@
(
_user_specified_nameresult_grads_8:JF

_output_shapes
:@
(
_user_specified_nameresult_grads_7:OK

_output_shapes
:	�@
(
_user_specified_nameresult_grads_6:KG

_output_shapes	
:�
(
_user_specified_nameresult_grads_5:PL
 
_output_shapes
:
��
(
_user_specified_nameresult_grads_4:KG

_output_shapes	
:�
(
_user_specified_nameresult_grads_3:OK

_output_shapes
:	�
(
_user_specified_nameresult_grads_2:JF

_output_shapes
:
(
_user_specified_nameresult_grads_1:N J

_output_shapes

:
(
_user_specified_nameresult_grads_0
�
�
"__inference_signature_wrapper_1858
x
y
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11:	 

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:
��

unknown_22:
��

unknown_23:	�

unknown_24:	�

unknown_25:	�@

unknown_26:	�@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:@

unknown_31:

unknown_32:

unknown_33: 

unknown_34: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *
fR
__inference_train_1714^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$% 

_user_specified_name1852:$$ 

_user_specified_name1850:$# 

_user_specified_name1848:$" 

_user_specified_name1846:$! 

_user_specified_name1844:$  

_user_specified_name1842:$ 

_user_specified_name1840:$ 

_user_specified_name1838:$ 

_user_specified_name1836:$ 

_user_specified_name1834:$ 

_user_specified_name1832:$ 

_user_specified_name1830:$ 

_user_specified_name1828:$ 

_user_specified_name1826:$ 

_user_specified_name1824:$ 

_user_specified_name1822:$ 

_user_specified_name1820:$ 

_user_specified_name1818:$ 

_user_specified_name1816:$ 

_user_specified_name1814:$ 

_user_specified_name1812:$ 

_user_specified_name1810:$ 

_user_specified_name1808:$ 

_user_specified_name1806:$ 

_user_specified_name1804:$ 

_user_specified_name1802:$ 

_user_specified_name1800:$
 

_user_specified_name1798:$	 

_user_specified_name1796:$ 

_user_specified_name1794:$ 

_user_specified_name1792:$ 

_user_specified_name1790:$ 

_user_specified_name1788:$ 

_user_specified_name1786:$ 

_user_specified_name1784:$ 

_user_specified_name1782:JF
'
_output_shapes
:���������

_user_specified_namey:J F
'
_output_shapes
:���������

_user_specified_namex
�

�
A__inference_dense_3_layer_call_and_return_conditional_losses_2935

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
 __inference__traced_restore_3841
file_prefix/
assignvariableop_dense_kernel:+
assignvariableop_1_dense_bias:4
!assignvariableop_2_dense_1_kernel:	�.
assignvariableop_3_dense_1_bias:	�5
!assignvariableop_4_dense_2_kernel:
��.
assignvariableop_5_dense_2_bias:	�4
!assignvariableop_6_dense_3_kernel:	�@-
assignvariableop_7_dense_3_bias:@3
!assignvariableop_8_dense_4_kernel:@-
assignvariableop_9_dense_4_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: 9
'assignvariableop_12_adam_m_dense_kernel:9
'assignvariableop_13_adam_v_dense_kernel:3
%assignvariableop_14_adam_m_dense_bias:3
%assignvariableop_15_adam_v_dense_bias:<
)assignvariableop_16_adam_m_dense_1_kernel:	�<
)assignvariableop_17_adam_v_dense_1_kernel:	�6
'assignvariableop_18_adam_m_dense_1_bias:	�6
'assignvariableop_19_adam_v_dense_1_bias:	�=
)assignvariableop_20_adam_m_dense_2_kernel:
��=
)assignvariableop_21_adam_v_dense_2_kernel:
��6
'assignvariableop_22_adam_m_dense_2_bias:	�6
'assignvariableop_23_adam_v_dense_2_bias:	�<
)assignvariableop_24_adam_m_dense_3_kernel:	�@<
)assignvariableop_25_adam_v_dense_3_kernel:	�@5
'assignvariableop_26_adam_m_dense_3_bias:@5
'assignvariableop_27_adam_v_dense_3_bias:@;
)assignvariableop_28_adam_m_dense_4_kernel:@;
)assignvariableop_29_adam_v_dense_4_kernel:@5
'assignvariableop_30_adam_m_dense_4_bias:5
'assignvariableop_31_adam_v_dense_4_bias:%
assignvariableop_32_total_1: %
assignvariableop_33_count_1: #
assignvariableop_34_total: #
assignvariableop_35_count: 
identity_37��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6model/optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB9model/optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_m_dense_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_v_dense_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_adam_m_dense_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp%assignvariableop_15_adam_v_dense_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_dense_1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_dense_1_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_m_dense_1_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_v_dense_1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_m_dense_2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_v_dense_2_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_m_dense_2_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_v_dense_2_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_dense_3_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_dense_3_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_m_dense_3_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_v_dense_3_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_m_dense_4_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_v_dense_4_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_m_dense_4_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_v_dense_4_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_totalIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_countIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_37Identity_37:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%$!

_user_specified_namecount:%#!

_user_specified_nametotal:'"#
!
_user_specified_name	count_1:'!#
!
_user_specified_name	total_1:3 /
-
_user_specified_nameAdam/v/dense_4/bias:3/
-
_user_specified_nameAdam/m/dense_4/bias:51
/
_user_specified_nameAdam/v/dense_4/kernel:51
/
_user_specified_nameAdam/m/dense_4/kernel:3/
-
_user_specified_nameAdam/v/dense_3/bias:3/
-
_user_specified_nameAdam/m/dense_3/bias:51
/
_user_specified_nameAdam/v/dense_3/kernel:51
/
_user_specified_nameAdam/m/dense_3/kernel:3/
-
_user_specified_nameAdam/v/dense_2/bias:3/
-
_user_specified_nameAdam/m/dense_2/bias:51
/
_user_specified_nameAdam/v/dense_2/kernel:51
/
_user_specified_nameAdam/m/dense_2/kernel:3/
-
_user_specified_nameAdam/v/dense_1/bias:3/
-
_user_specified_nameAdam/m/dense_1/bias:51
/
_user_specified_nameAdam/v/dense_1/kernel:51
/
_user_specified_nameAdam/m/dense_1/kernel:1-
+
_user_specified_nameAdam/v/dense/bias:1-
+
_user_specified_nameAdam/m/dense/bias:3/
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,
(
&
_user_specified_namedense_4/bias:.	*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_3254

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_3100

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_dropout_1_layer_call_fn_3242

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3100a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�(
__inference_train_2785
x
y<
*model_dense_matmul_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:?
,model_dense_1_matmul_readvariableop_resource:	�<
-model_dense_1_biasadd_readvariableop_resource:	�@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�?
,model_dense_3_matmul_readvariableop_resource:	�@;
-model_dense_3_biasadd_readvariableop_resource:@>
,model_dense_4_matmul_readvariableop_resource:@;
-model_dense_4_biasadd_readvariableop_resource:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: &
adam_readvariableop_resource:	 (
adam_readvariableop_1_resource: 4
"adam_sub_2_readvariableop_resource:4
"adam_sub_3_readvariableop_resource:0
"adam_sub_6_readvariableop_resource:0
"adam_sub_7_readvariableop_resource:6
#adam_sub_10_readvariableop_resource:	�6
#adam_sub_11_readvariableop_resource:	�2
#adam_sub_14_readvariableop_resource:	�2
#adam_sub_15_readvariableop_resource:	�7
#adam_sub_18_readvariableop_resource:
��7
#adam_sub_19_readvariableop_resource:
��2
#adam_sub_22_readvariableop_resource:	�2
#adam_sub_23_readvariableop_resource:	�6
#adam_sub_26_readvariableop_resource:	�@6
#adam_sub_27_readvariableop_resource:	�@1
#adam_sub_30_readvariableop_resource:@1
#adam_sub_31_readvariableop_resource:@5
#adam_sub_34_readvariableop_resource:@5
#adam_sub_35_readvariableop_resource:@1
#adam_sub_38_readvariableop_resource:1
#adam_sub_39_readvariableop_resource:(
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: 
identity_12
identity_13��Adam/AssignAddVariableOp�Adam/AssignAddVariableOp_1�Adam/AssignAddVariableOp_10�Adam/AssignAddVariableOp_11�Adam/AssignAddVariableOp_12�Adam/AssignAddVariableOp_13�Adam/AssignAddVariableOp_14�Adam/AssignAddVariableOp_15�Adam/AssignAddVariableOp_16�Adam/AssignAddVariableOp_17�Adam/AssignAddVariableOp_18�Adam/AssignAddVariableOp_19�Adam/AssignAddVariableOp_2�Adam/AssignAddVariableOp_20�Adam/AssignAddVariableOp_3�Adam/AssignAddVariableOp_4�Adam/AssignAddVariableOp_5�Adam/AssignAddVariableOp_6�Adam/AssignAddVariableOp_7�Adam/AssignAddVariableOp_8�Adam/AssignAddVariableOp_9�Adam/AssignSubVariableOp�Adam/AssignSubVariableOp_1�Adam/AssignSubVariableOp_2�Adam/AssignSubVariableOp_3�Adam/AssignSubVariableOp_4�Adam/AssignSubVariableOp_5�Adam/AssignSubVariableOp_6�Adam/AssignSubVariableOp_7�Adam/AssignSubVariableOp_8�Adam/AssignSubVariableOp_9�Adam/ReadVariableOp�Adam/ReadVariableOp_1�Adam/ReadVariableOp_10�Adam/ReadVariableOp_11�Adam/ReadVariableOp_12�Adam/ReadVariableOp_13�Adam/ReadVariableOp_14�Adam/ReadVariableOp_15�Adam/ReadVariableOp_16�Adam/ReadVariableOp_17�Adam/ReadVariableOp_18�Adam/ReadVariableOp_19�Adam/ReadVariableOp_2�Adam/ReadVariableOp_20�Adam/ReadVariableOp_21�Adam/ReadVariableOp_22�Adam/ReadVariableOp_23�Adam/ReadVariableOp_24�Adam/ReadVariableOp_25�Adam/ReadVariableOp_26�Adam/ReadVariableOp_27�Adam/ReadVariableOp_28�Adam/ReadVariableOp_29�Adam/ReadVariableOp_3�Adam/ReadVariableOp_4�Adam/ReadVariableOp_5�Adam/ReadVariableOp_6�Adam/ReadVariableOp_7�Adam/ReadVariableOp_8�Adam/ReadVariableOp_9�Adam/Sqrt_1/ReadVariableOp�Adam/Sqrt_11/ReadVariableOp�Adam/Sqrt_13/ReadVariableOp�Adam/Sqrt_15/ReadVariableOp�Adam/Sqrt_17/ReadVariableOp�Adam/Sqrt_19/ReadVariableOp�Adam/Sqrt_3/ReadVariableOp�Adam/Sqrt_5/ReadVariableOp�Adam/Sqrt_7/ReadVariableOp�Adam/Sqrt_9/ReadVariableOp�Adam/sub_10/ReadVariableOp�Adam/sub_11/ReadVariableOp�Adam/sub_14/ReadVariableOp�Adam/sub_15/ReadVariableOp�Adam/sub_18/ReadVariableOp�Adam/sub_19/ReadVariableOp�Adam/sub_2/ReadVariableOp�Adam/sub_22/ReadVariableOp�Adam/sub_23/ReadVariableOp�Adam/sub_26/ReadVariableOp�Adam/sub_27/ReadVariableOp�Adam/sub_3/ReadVariableOp�Adam/sub_30/ReadVariableOp�Adam/sub_31/ReadVariableOp�Adam/sub_34/ReadVariableOp�Adam/sub_35/ReadVariableOp�Adam/sub_38/ReadVariableOp�Adam/sub_39/ReadVariableOp�Adam/sub_6/ReadVariableOp�Adam/sub_7/ReadVariableOp�AssignAddVariableOp�AssignAddVariableOp_1�AssignAddVariableOp_2�AssignAddVariableOp_3�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1�div_no_nan_1/ReadVariableOp�div_no_nan_1/ReadVariableOp_1�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0|
model/dense/MatMulMatMulx)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������`
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout/dropout/MulMul model/dense_1/Relu:activations:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������y
model/dropout/dropout/ShapeShape model/dense_1/Relu:activations:0*
T0*
_output_shapes
::���
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0i
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������b
model/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
model/dropout/dropout/SelectV2SelectV2&model/dropout/dropout/GreaterEqual:z:0model/dropout/dropout/Mul:z:0&model/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul'model/dropout/dropout/SelectV2:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������b
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_1/dropout/MulMul model/dense_2/Relu:activations:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������{
model/dropout_1/dropout/ShapeShape model/dense_2/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0k
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������d
model/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_1/dropout/SelectV2SelectV2(model/dropout_1/dropout/GreaterEqual:z:0model/dropout_1/dropout/Mul:z:0(model/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_3/MatMulMatMul)model/dropout_1/dropout/SelectV2:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@b
model/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_2/dropout/MulMul model/dense_3/Relu:activations:0&model/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@{
model/dropout_2/dropout/ShapeShape model/dense_3/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_2/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0k
&model/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_2/dropout/GreaterEqualGreaterEqual=model/dropout_2/dropout/random_uniform/RandomUniform:output:0/model/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@d
model/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_2/dropout/SelectV2SelectV2(model/dropout_2/dropout/GreaterEqual:z:0model/dropout_2/dropout/Mul:z:0(model/dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_4/MatMulMatMul)model/dropout_2/dropout/SelectV2:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_4/SigmoidSigmoidmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������^
binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,binary_crossentropy/logistic_loss/zeros_like	ZerosLikemodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqualmodel/dense_4/BiasAdd:output:00binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*'
_output_shapes
:����������
(binary_crossentropy/logistic_loss/SelectSelect2binary_crossentropy/logistic_loss/GreaterEqual:z:0model/dense_4/BiasAdd:output:00binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*'
_output_shapes
:���������~
%binary_crossentropy/logistic_loss/NegNegmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
*binary_crossentropy/logistic_loss/Select_1Select2binary_crossentropy/logistic_loss/GreaterEqual:z:0)binary_crossentropy/logistic_loss/Neg:y:0model/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
%binary_crossentropy/logistic_loss/mulMulmodel/dense_4/BiasAdd:output:0y*
T0*'
_output_shapes
:����������
%binary_crossentropy/logistic_loss/subSub1binary_crossentropy/logistic_loss/Select:output:0)binary_crossentropy/logistic_loss/mul:z:0*
T0*'
_output_shapes
:����������
%binary_crossentropy/logistic_loss/ExpExp3binary_crossentropy/logistic_loss/Select_1:output:0*
T0*'
_output_shapes
:����������
'binary_crossentropy/logistic_loss/Log1pLog1p)binary_crossentropy/logistic_loss/Exp:y:0*
T0*'
_output_shapes
:����������
!binary_crossentropy/logistic_lossAddV2)binary_crossentropy/logistic_loss/sub:z:0+binary_crossentropy/logistic_loss/Log1p:y:0*
T0*'
_output_shapes
:���������u
*binary_crossentropy/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
binary_crossentropy/MeanMean%binary_crossentropy/logistic_loss:z:03binary_crossentropy/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������l
'binary_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%binary_crossentropy/weighted_loss/MulMul!binary_crossentropy/Mean:output:00binary_crossentropy/weighted_loss/Const:output:0*
T0*#
_output_shapes
:���������s
)binary_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%binary_crossentropy/weighted_loss/SumSum)binary_crossentropy/weighted_loss/Mul:z:02binary_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: �
.binary_crossentropy/weighted_loss/num_elementsSize)binary_crossentropy/weighted_loss/Mul:z:0*
T0*
_output_shapes
: �
3binary_crossentropy/weighted_loss/num_elements/CastCast7binary_crossentropy/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: h
&binary_crossentropy/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : o
-binary_crossentropy/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : o
-binary_crossentropy/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
'binary_crossentropy/weighted_loss/rangeRange6binary_crossentropy/weighted_loss/range/start:output:0/binary_crossentropy/weighted_loss/Rank:output:06binary_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: �
'binary_crossentropy/weighted_loss/Sum_1Sum.binary_crossentropy/weighted_loss/Sum:output:00binary_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: �
'binary_crossentropy/weighted_loss/valueDivNoNan0binary_crossentropy/weighted_loss/Sum_1:output:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: D
ShapeShapey*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: b
MulMul+binary_crossentropy/weighted_loss/value:z:0Cast:y:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: l
SumSumMul:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: I
Sum_1SumCast:y:0range_1:output:0*
T0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum_1:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype0I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nanDivNoNanones:output:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
9gradient_tape/binary_crossentropy/weighted_loss/value/NegNeg0binary_crossentropy/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: �
Bgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_1DivNoNan=gradient_tape/binary_crossentropy/weighted_loss/value/Neg:y:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
Bgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_2DivNoNanFgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_1:z:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
9gradient_tape/binary_crossentropy/weighted_loss/value/mulMulones:output:0Fgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: ~
;gradient_tape/binary_crossentropy/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB �
=gradient_tape/binary_crossentropy/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
=gradient_tape/binary_crossentropy/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB �
?gradient_tape/binary_crossentropy/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB �
7gradient_tape/binary_crossentropy/weighted_loss/ReshapeReshapeDgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan:z:0Hgradient_tape/binary_crossentropy/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: x
5gradient_tape/binary_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB �
4gradient_tape/binary_crossentropy/weighted_loss/TileTile@gradient_tape/binary_crossentropy/weighted_loss/Reshape:output:0>gradient_tape/binary_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: �
?gradient_tape/binary_crossentropy/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
9gradient_tape/binary_crossentropy/weighted_loss/Reshape_1Reshape=gradient_tape/binary_crossentropy/weighted_loss/Tile:output:0Hgradient_tape/binary_crossentropy/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:�
5gradient_tape/binary_crossentropy/weighted_loss/ShapeShape)binary_crossentropy/weighted_loss/Mul:z:0*
T0*
_output_shapes
::���
6gradient_tape/binary_crossentropy/weighted_loss/Tile_1TileBgradient_tape/binary_crossentropy/weighted_loss/Reshape_1:output:0>gradient_tape/binary_crossentropy/weighted_loss/Shape:output:0*
T0*#
_output_shapes
:����������
3gradient_tape/binary_crossentropy/weighted_loss/MulMul?gradient_tape/binary_crossentropy/weighted_loss/Tile_1:output:00binary_crossentropy/weighted_loss/Const:output:0*
T0*#
_output_shapes
:����������
'gradient_tape/binary_crossentropy/ShapeShape%binary_crossentropy/logistic_loss:z:0*
T0*
_output_shapes
::���
&gradient_tape/binary_crossentropy/SizeConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B :�
%gradient_tape/binary_crossentropy/addAddV23binary_crossentropy/Mean/reduction_indices:output:0/gradient_tape/binary_crossentropy/Size:output:0*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: �
%gradient_tape/binary_crossentropy/modFloorMod)gradient_tape/binary_crossentropy/add:z:0/gradient_tape/binary_crossentropy/Size:output:0*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: �
)gradient_tape/binary_crossentropy/Shape_1Const*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
valueB �
-gradient_tape/binary_crossentropy/range/startConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B : �
-gradient_tape/binary_crossentropy/range/deltaConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B :�
'gradient_tape/binary_crossentropy/rangeRange6gradient_tape/binary_crossentropy/range/start:output:0/gradient_tape/binary_crossentropy/Size:output:06gradient_tape/binary_crossentropy/range/delta:output:0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
:�
,gradient_tape/binary_crossentropy/ones/ConstConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B :�
&gradient_tape/binary_crossentropy/onesFill2gradient_tape/binary_crossentropy/Shape_1:output:05gradient_tape/binary_crossentropy/ones/Const:output:0*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: �
/gradient_tape/binary_crossentropy/DynamicStitchDynamicStitch0gradient_tape/binary_crossentropy/range:output:0)gradient_tape/binary_crossentropy/mod:z:00gradient_tape/binary_crossentropy/Shape:output:0/gradient_tape/binary_crossentropy/ones:output:0*
N*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
:�
)gradient_tape/binary_crossentropy/ReshapeReshape7gradient_tape/binary_crossentropy/weighted_loss/Mul:z:08gradient_tape/binary_crossentropy/DynamicStitch:merged:0*
T0*0
_output_shapes
:�������������������
-gradient_tape/binary_crossentropy/BroadcastToBroadcastTo2gradient_tape/binary_crossentropy/Reshape:output:00gradient_tape/binary_crossentropy/Shape:output:0*
T0*'
_output_shapes
:����������
)gradient_tape/binary_crossentropy/Shape_2Shape%binary_crossentropy/logistic_loss:z:0*
T0*
_output_shapes
::��j
(gradient_tape/binary_crossentropy/Size_1Const*
_output_shapes
: *
dtype0*
value	B :�
'gradient_tape/binary_crossentropy/add_1AddV23binary_crossentropy/Mean/reduction_indices:output:01gradient_tape/binary_crossentropy/Size_1:output:0*
T0*
_output_shapes
: �
'gradient_tape/binary_crossentropy/mod_1FloorMod+gradient_tape/binary_crossentropy/add_1:z:01gradient_tape/binary_crossentropy/Size_1:output:0*
T0*
_output_shapes
: q
/gradient_tape/binary_crossentropy/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*gradient_tape/binary_crossentropy/GatherV2GatherV22gradient_tape/binary_crossentropy/Shape_2:output:0+gradient_tape/binary_crossentropy/mod_1:z:08gradient_tape/binary_crossentropy/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: h
&gradient_tape/binary_crossentropy/RankConst*
_output_shapes
: *
dtype0*
value	B : q
/gradient_tape/binary_crossentropy/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : q
/gradient_tape/binary_crossentropy/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
)gradient_tape/binary_crossentropy/range_1Range8gradient_tape/binary_crossentropy/range_1/start:output:0/gradient_tape/binary_crossentropy/Rank:output:08gradient_tape/binary_crossentropy/range_1/delta:output:0*
_output_shapes
: �
&gradient_tape/binary_crossentropy/ProdProd3gradient_tape/binary_crossentropy/GatherV2:output:02gradient_tape/binary_crossentropy/range_1:output:0*
T0*
_output_shapes
: �
&gradient_tape/binary_crossentropy/CastCast/gradient_tape/binary_crossentropy/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
)gradient_tape/binary_crossentropy/truedivRealDiv6gradient_tape/binary_crossentropy/BroadcastTo:output:0*gradient_tape/binary_crossentropy/Cast:y:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/ShapeShape)binary_crossentropy/logistic_loss/sub:z:0*
T0*
_output_shapes
::���
7gradient_tape/binary_crossentropy/logistic_loss/Shape_1Shape+binary_crossentropy/logistic_loss/Log1p:y:0*
T0*
_output_shapes
::���
Egradient_tape/binary_crossentropy/logistic_loss/BroadcastGradientArgsBroadcastGradientArgs>gradient_tape/binary_crossentropy/logistic_loss/Shape:output:0@gradient_tape/binary_crossentropy/logistic_loss/Shape_1:output:0*2
_output_shapes 
:���������:����������
3gradient_tape/binary_crossentropy/logistic_loss/SumSum-gradient_tape/binary_crossentropy/truediv:z:0Jgradient_tape/binary_crossentropy/logistic_loss/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
7gradient_tape/binary_crossentropy/logistic_loss/ReshapeReshape<gradient_tape/binary_crossentropy/logistic_loss/Sum:output:0>gradient_tape/binary_crossentropy/logistic_loss/Shape:output:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/Sum_1Sum-gradient_tape/binary_crossentropy/truediv:z:0Jgradient_tape/binary_crossentropy/logistic_loss/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
9gradient_tape/binary_crossentropy/logistic_loss/Reshape_1Reshape>gradient_tape/binary_crossentropy/logistic_loss/Sum_1:output:0@gradient_tape/binary_crossentropy/logistic_loss/Shape_1:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:����������
7gradient_tape/binary_crossentropy/logistic_loss/sub/NegNeg@gradient_tape/binary_crossentropy/logistic_loss/Reshape:output:0*
T0*'
_output_shapes
:����������
9gradient_tape/binary_crossentropy/logistic_loss/sub/ShapeShape1binary_crossentropy/logistic_loss/Select:output:0*
T0*
_output_shapes
::���
;gradient_tape/binary_crossentropy/logistic_loss/sub/Shape_1Shape)binary_crossentropy/logistic_loss/mul:z:0*
T0*
_output_shapes
::���
Igradient_tape/binary_crossentropy/logistic_loss/sub/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/binary_crossentropy/logistic_loss/sub/Shape:output:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Shape_1:output:0*2
_output_shapes 
:���������:����������
7gradient_tape/binary_crossentropy/logistic_loss/sub/SumSum@gradient_tape/binary_crossentropy/logistic_loss/Reshape:output:0Ngradient_tape/binary_crossentropy/logistic_loss/sub/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
;gradient_tape/binary_crossentropy/logistic_loss/sub/ReshapeReshape@gradient_tape/binary_crossentropy/logistic_loss/sub/Sum:output:0Bgradient_tape/binary_crossentropy/logistic_loss/sub/Shape:output:0*
T0*'
_output_shapes
:����������
9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1Sum;gradient_tape/binary_crossentropy/logistic_loss/sub/Neg:y:0Ngradient_tape/binary_crossentropy/logistic_loss/sub/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
=gradient_tape/binary_crossentropy/logistic_loss/sub/Reshape_1ReshapeBgradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1:output:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Shape_1:output:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/add/xConst:^gradient_tape/binary_crossentropy/logistic_loss/Reshape_1*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3gradient_tape/binary_crossentropy/logistic_loss/addAddV2>gradient_tape/binary_crossentropy/logistic_loss/add/x:output:0)binary_crossentropy/logistic_loss/Exp:y:0*
T0*'
_output_shapes
:����������
:gradient_tape/binary_crossentropy/logistic_loss/Reciprocal
Reciprocal7gradient_tape/binary_crossentropy/logistic_loss/add:z:0*
T0*'
_output_shapes
:����������
3gradient_tape/binary_crossentropy/logistic_loss/mulMulBgradient_tape/binary_crossentropy/logistic_loss/Reshape_1:output:0>gradient_tape/binary_crossentropy/logistic_loss/Reciprocal:y:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:����������
:gradient_tape/binary_crossentropy/logistic_loss/zeros_like	ZerosLikemodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
6gradient_tape/binary_crossentropy/logistic_loss/SelectSelect2binary_crossentropy/logistic_loss/GreaterEqual:z:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Reshape:output:0>gradient_tape/binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*'
_output_shapes
:����������
8gradient_tape/binary_crossentropy/logistic_loss/Select_1Select2binary_crossentropy/logistic_loss/GreaterEqual:z:0>gradient_tape/binary_crossentropy/logistic_loss/zeros_like:y:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Reshape:output:0*
T0*'
_output_shapes
:����������
7gradient_tape/binary_crossentropy/logistic_loss/mul/MulMulFgradient_tape/binary_crossentropy/logistic_loss/sub/Reshape_1:output:0y*
T0*'
_output_shapes
:����������
9gradient_tape/binary_crossentropy/logistic_loss/mul/ShapeShapemodel/dense_4/BiasAdd:output:0*
T0*
_output_shapes
::��z
;gradient_tape/binary_crossentropy/logistic_loss/mul/Shape_1Shapey*
T0*
_output_shapes
::���
Igradient_tape/binary_crossentropy/logistic_loss/mul/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/binary_crossentropy/logistic_loss/mul/Shape:output:0Dgradient_tape/binary_crossentropy/logistic_loss/mul/Shape_1:output:0*2
_output_shapes 
:���������:����������
7gradient_tape/binary_crossentropy/logistic_loss/mul/SumSum;gradient_tape/binary_crossentropy/logistic_loss/mul/Mul:z:0Ngradient_tape/binary_crossentropy/logistic_loss/mul/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
;gradient_tape/binary_crossentropy/logistic_loss/mul/ReshapeReshape@gradient_tape/binary_crossentropy/logistic_loss/mul/Sum:output:0Bgradient_tape/binary_crossentropy/logistic_loss/mul/Shape:output:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/mul_1Mul7gradient_tape/binary_crossentropy/logistic_loss/mul:z:0)binary_crossentropy/logistic_loss/Exp:y:0*
T0*'
_output_shapes
:����������
<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1	ZerosLike)binary_crossentropy/logistic_loss/Neg:y:0*
T0*'
_output_shapes
:����������
8gradient_tape/binary_crossentropy/logistic_loss/Select_2Select2binary_crossentropy/logistic_loss/GreaterEqual:z:09gradient_tape/binary_crossentropy/logistic_loss/mul_1:z:0@gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1:y:0*
T0*'
_output_shapes
:����������
8gradient_tape/binary_crossentropy/logistic_loss/Select_3Select2binary_crossentropy/logistic_loss/GreaterEqual:z:0@gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1:y:09gradient_tape/binary_crossentropy/logistic_loss/mul_1:z:0*
T0*'
_output_shapes
:����������
3gradient_tape/binary_crossentropy/logistic_loss/NegNegAgradient_tape/binary_crossentropy/logistic_loss/Select_2:output:0*
T0*'
_output_shapes
:����������
AddNAddN?gradient_tape/binary_crossentropy/logistic_loss/Select:output:0Dgradient_tape/binary_crossentropy/logistic_loss/mul/Reshape:output:0Agradient_tape/binary_crossentropy/logistic_loss/Select_3:output:07gradient_tape/binary_crossentropy/logistic_loss/Neg:y:0*
N*
T0*'
_output_shapes
:���������o
/gradient_tape/model/dense_4/BiasAdd/BiasAddGradBiasAddGrad
AddN:sum:0*
T0*
_output_shapes
:�
)gradient_tape/model/dense_4/MatMul/MatMulMatMul
AddN:sum:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_4/MatMul/MatMul_1MatMul)model/dropout_2/dropout/SelectV2:output:0
AddN:sum:0*
T0*
_output_shapes

:@*
grad_b(*
transpose_a(p
+gradient_tape/model/dropout_2/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.gradient_tape/model/dropout_2/dropout/SelectV2SelectV2(model/dropout_2/dropout/GreaterEqual:z:03gradient_tape/model/dense_4/MatMul/MatMul:product:04gradient_tape/model/dropout_2/dropout/zeros:output:0*
T0*'
_output_shapes
:���������@�
0gradient_tape/model/dropout_2/dropout/SelectV2_1SelectV2(model/dropout_2/dropout/GreaterEqual:z:04gradient_tape/model/dropout_2/dropout/zeros:output:03gradient_tape/model/dense_4/MatMul/MatMul:product:0*
T0*'
_output_shapes
:���������@�
+gradient_tape/model/dropout_2/dropout/ShapeShapemodel/dropout_2/dropout/Mul:z:0*
T0*
_output_shapes
::���
-gradient_tape/model/dropout_2/dropout/Shape_1Shape)model/dropout_2/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
;gradient_tape/model/dropout_2/dropout/BroadcastGradientArgsBroadcastGradientArgs4gradient_tape/model/dropout_2/dropout/Shape:output:06gradient_tape/model/dropout_2/dropout/Shape_1:output:0*2
_output_shapes 
:���������:����������
)gradient_tape/model/dropout_2/dropout/SumSum7gradient_tape/model/dropout_2/dropout/SelectV2:output:0@gradient_tape/model/dropout_2/dropout/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
-gradient_tape/model/dropout_2/dropout/ReshapeReshape2gradient_tape/model/dropout_2/dropout/Sum:output:04gradient_tape/model/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@p
-gradient_tape/model/dropout_2/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB �
-gradient_tape/model/dropout_2/dropout/Shape_3Shape)model/dropout_2/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
=gradient_tape/model/dropout_2/dropout/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
+gradient_tape/model/dropout_2/dropout/Sum_1Sum9gradient_tape/model/dropout_2/dropout/SelectV2_1:output:0Fgradient_tape/model/dropout_2/dropout/Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
/gradient_tape/model/dropout_2/dropout/Reshape_1Reshape4gradient_tape/model/dropout_2/dropout/Sum_1:output:06gradient_tape/model/dropout_2/dropout/Shape_2:output:0*
T0*
_output_shapes
: �
)gradient_tape/model/dropout_2/dropout/MulMul6gradient_tape/model/dropout_2/dropout/Reshape:output:0&model/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@�
$gradient_tape/model/dense_3/ReluGradReluGrad-gradient_tape/model/dropout_2/dropout/Mul:z:0 model/dense_3/Relu:activations:0*
T0*'
_output_shapes
:���������@�
/gradient_tape/model/dense_3/BiasAdd/BiasAddGradBiasAddGrad0gradient_tape/model/dense_3/ReluGrad:backprops:0*
T0*
_output_shapes
:@�
)gradient_tape/model/dense_3/MatMul/MatMulMatMul0gradient_tape/model/dense_3/ReluGrad:backprops:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_3/MatMul/MatMul_1MatMul)model/dropout_1/dropout/SelectV2:output:00gradient_tape/model/dense_3/ReluGrad:backprops:0*
T0*
_output_shapes
:	�@*
grad_b(*
transpose_a(p
+gradient_tape/model/dropout_1/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.gradient_tape/model/dropout_1/dropout/SelectV2SelectV2(model/dropout_1/dropout/GreaterEqual:z:03gradient_tape/model/dense_3/MatMul/MatMul:product:04gradient_tape/model/dropout_1/dropout/zeros:output:0*
T0*(
_output_shapes
:�����������
0gradient_tape/model/dropout_1/dropout/SelectV2_1SelectV2(model/dropout_1/dropout/GreaterEqual:z:04gradient_tape/model/dropout_1/dropout/zeros:output:03gradient_tape/model/dense_3/MatMul/MatMul:product:0*
T0*(
_output_shapes
:�����������
+gradient_tape/model/dropout_1/dropout/ShapeShapemodel/dropout_1/dropout/Mul:z:0*
T0*
_output_shapes
::���
-gradient_tape/model/dropout_1/dropout/Shape_1Shape)model/dropout_1/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
;gradient_tape/model/dropout_1/dropout/BroadcastGradientArgsBroadcastGradientArgs4gradient_tape/model/dropout_1/dropout/Shape:output:06gradient_tape/model/dropout_1/dropout/Shape_1:output:0*2
_output_shapes 
:���������:����������
)gradient_tape/model/dropout_1/dropout/SumSum7gradient_tape/model/dropout_1/dropout/SelectV2:output:0@gradient_tape/model/dropout_1/dropout/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
-gradient_tape/model/dropout_1/dropout/ReshapeReshape2gradient_tape/model/dropout_1/dropout/Sum:output:04gradient_tape/model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������p
-gradient_tape/model/dropout_1/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB �
-gradient_tape/model/dropout_1/dropout/Shape_3Shape)model/dropout_1/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
=gradient_tape/model/dropout_1/dropout/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
+gradient_tape/model/dropout_1/dropout/Sum_1Sum9gradient_tape/model/dropout_1/dropout/SelectV2_1:output:0Fgradient_tape/model/dropout_1/dropout/Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
/gradient_tape/model/dropout_1/dropout/Reshape_1Reshape4gradient_tape/model/dropout_1/dropout/Sum_1:output:06gradient_tape/model/dropout_1/dropout/Shape_2:output:0*
T0*
_output_shapes
: �
)gradient_tape/model/dropout_1/dropout/MulMul6gradient_tape/model/dropout_1/dropout/Reshape:output:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$gradient_tape/model/dense_2/ReluGradReluGrad-gradient_tape/model/dropout_1/dropout/Mul:z:0 model/dense_2/Relu:activations:0*
T0*(
_output_shapes
:�����������
/gradient_tape/model/dense_2/BiasAdd/BiasAddGradBiasAddGrad0gradient_tape/model/dense_2/ReluGrad:backprops:0*
T0*
_output_shapes	
:��
)gradient_tape/model/dense_2/MatMul/MatMulMatMul0gradient_tape/model/dense_2/ReluGrad:backprops:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_2/MatMul/MatMul_1MatMul'model/dropout/dropout/SelectV2:output:00gradient_tape/model/dense_2/ReluGrad:backprops:0*
T0* 
_output_shapes
:
��*
grad_b(*
transpose_a(n
)gradient_tape/model/dropout/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,gradient_tape/model/dropout/dropout/SelectV2SelectV2&model/dropout/dropout/GreaterEqual:z:03gradient_tape/model/dense_2/MatMul/MatMul:product:02gradient_tape/model/dropout/dropout/zeros:output:0*
T0*(
_output_shapes
:�����������
.gradient_tape/model/dropout/dropout/SelectV2_1SelectV2&model/dropout/dropout/GreaterEqual:z:02gradient_tape/model/dropout/dropout/zeros:output:03gradient_tape/model/dense_2/MatMul/MatMul:product:0*
T0*(
_output_shapes
:�����������
)gradient_tape/model/dropout/dropout/ShapeShapemodel/dropout/dropout/Mul:z:0*
T0*
_output_shapes
::���
+gradient_tape/model/dropout/dropout/Shape_1Shape'model/dropout/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
9gradient_tape/model/dropout/dropout/BroadcastGradientArgsBroadcastGradientArgs2gradient_tape/model/dropout/dropout/Shape:output:04gradient_tape/model/dropout/dropout/Shape_1:output:0*2
_output_shapes 
:���������:����������
'gradient_tape/model/dropout/dropout/SumSum5gradient_tape/model/dropout/dropout/SelectV2:output:0>gradient_tape/model/dropout/dropout/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
+gradient_tape/model/dropout/dropout/ReshapeReshape0gradient_tape/model/dropout/dropout/Sum:output:02gradient_tape/model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������n
+gradient_tape/model/dropout/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB �
+gradient_tape/model/dropout/dropout/Shape_3Shape'model/dropout/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
;gradient_tape/model/dropout/dropout/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
)gradient_tape/model/dropout/dropout/Sum_1Sum7gradient_tape/model/dropout/dropout/SelectV2_1:output:0Dgradient_tape/model/dropout/dropout/Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-gradient_tape/model/dropout/dropout/Reshape_1Reshape2gradient_tape/model/dropout/dropout/Sum_1:output:04gradient_tape/model/dropout/dropout/Shape_2:output:0*
T0*
_output_shapes
: �
'gradient_tape/model/dropout/dropout/MulMul4gradient_tape/model/dropout/dropout/Reshape:output:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$gradient_tape/model/dense_1/ReluGradReluGrad+gradient_tape/model/dropout/dropout/Mul:z:0 model/dense_1/Relu:activations:0*
T0*(
_output_shapes
:�����������
/gradient_tape/model/dense_1/BiasAdd/BiasAddGradBiasAddGrad0gradient_tape/model/dense_1/ReluGrad:backprops:0*
T0*
_output_shapes	
:��
)gradient_tape/model/dense_1/MatMul/MatMulMatMul0gradient_tape/model/dense_1/ReluGrad:backprops:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_1/MatMul/MatMul_1MatMulmodel/dense/Relu:activations:00gradient_tape/model/dense_1/ReluGrad:backprops:0*
T0*
_output_shapes
:	�*
grad_b(*
transpose_a(�
"gradient_tape/model/dense/ReluGradReluGrad3gradient_tape/model/dense_1/MatMul/MatMul:product:0model/dense/Relu:activations:0*
T0*'
_output_shapes
:����������
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGrad.gradient_tape/model/dense/ReluGrad:backprops:0*
T0*
_output_shapes
:�
'gradient_tape/model/dense/MatMul/MatMulMatMulx.gradient_tape/model/dense/ReluGrad:backprops:0*
T0*
_output_shapes

:*
grad_b(*
transpose_a(p
IdentityIdentity1gradient_tape/model/dense/MatMul/MatMul:product:0*
T0*
_output_shapes

:s

Identity_1Identity6gradient_tape/model/dense/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:w

Identity_2Identity5gradient_tape/model/dense_1/MatMul/MatMul_1:product:0*
T0*
_output_shapes
:	�v

Identity_3Identity8gradient_tape/model/dense_1/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:�x

Identity_4Identity5gradient_tape/model/dense_2/MatMul/MatMul_1:product:0*
T0* 
_output_shapes
:
��v

Identity_5Identity8gradient_tape/model/dense_2/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:�w

Identity_6Identity5gradient_tape/model/dense_3/MatMul/MatMul_1:product:0*
T0*
_output_shapes
:	�@u

Identity_7Identity8gradient_tape/model/dense_3/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:@v

Identity_8Identity5gradient_tape/model/dense_4/MatMul/MatMul_1:product:0*
T0*
_output_shapes

:@u

Identity_9Identity8gradient_tape/model/dense_4/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:�
	IdentityN	IdentityN1gradient_tape/model/dense/MatMul/MatMul:product:06gradient_tape/model/dense/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_1/MatMul/MatMul_1:product:08gradient_tape/model/dense_1/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_2/MatMul/MatMul_1:product:08gradient_tape/model/dense_2/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_3/MatMul/MatMul_1:product:08gradient_tape/model/dense_3/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_4/MatMul/MatMul_1:product:08gradient_tape/model/dense_4/BiasAdd/BiasAddGrad:output:01gradient_tape/model/dense/MatMul/MatMul:product:06gradient_tape/model/dense/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_1/MatMul/MatMul_1:product:08gradient_tape/model/dense_1/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_2/MatMul/MatMul_1:product:08gradient_tape/model/dense_2/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_3/MatMul/MatMul_1:product:08gradient_tape/model/dense_3/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_4/MatMul/MatMul_1:product:08gradient_tape/model/dense_4/BiasAdd/BiasAddGrad:output:0*
T
2**
_gradient_op_typeCustomGradient-2388*�
_output_shapes�
�:::	�:�:
��:�:	�@:@:@::::	�:�:
��:�:	�@:@:@:h
Adam/ReadVariableOpReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	L

Adam/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd
Adam/addAddV2Adam/ReadVariableOp:value:0Adam/add/y:output:0*
T0	*
_output_shapes
: O
	Adam/CastCastAdam/add:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
Adam/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
Adam/PowPowAdam/Cast_1/x:output:0Adam/Cast:y:0*
T0*
_output_shapes
: R
Adam/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?Y

Adam/Pow_1PowAdam/Cast_2/x:output:0Adam/Cast:y:0*
T0*
_output_shapes
: O

Adam/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
Adam/subSubAdam/sub/x:output:0Adam/Pow_1:z:0*
T0*
_output_shapes
: @
	Adam/SqrtSqrtAdam/sub:z:0*
T0*
_output_shapes
: l
Adam/ReadVariableOp_1ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0^
Adam/mulMulAdam/ReadVariableOp_1:value:0Adam/Sqrt:y:0*
T0*
_output_shapes
: Q
Adam/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?W

Adam/sub_1SubAdam/sub_1/x:output:0Adam/Pow:z:0*
T0*
_output_shapes
: V
Adam/truedivRealDivAdam/mul:z:0Adam/sub_1:z:0*
T0*
_output_shapes
: |
Adam/sub_2/ReadVariableOpReadVariableOp"adam_sub_2_readvariableop_resource*
_output_shapes

:*
dtype0q

Adam/sub_2SubIdentityN:output:0!Adam/sub_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Q
Adam/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=a

Adam/mul_1MulAdam/sub_2:z:0Adam/mul_1/y:output:0*
T0*
_output_shapes

:�
Adam/AssignAddVariableOpAssignAddVariableOp"adam_sub_2_readvariableop_resourceAdam/mul_1:z:0^Adam/sub_2/ReadVariableOp*
_output_shapes
 *
dtype0R
Adam/SquareSquareIdentityN:output:0*
T0*
_output_shapes

:|
Adam/sub_3/ReadVariableOpReadVariableOp"adam_sub_3_readvariableop_resource*
_output_shapes

:*
dtype0n

Adam/sub_3SubAdam/Square:y:0!Adam/sub_3/ReadVariableOp:value:0*
T0*
_output_shapes

:Q
Adam/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a

Adam/mul_2MulAdam/sub_3:z:0Adam/mul_2/y:output:0*
T0*
_output_shapes

:�
Adam/AssignAddVariableOp_1AssignAddVariableOp"adam_sub_3_readvariableop_resourceAdam/mul_2:z:0^Adam/sub_3/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_2ReadVariableOp"adam_sub_2_readvariableop_resource^Adam/AssignAddVariableOp*
_output_shapes

:*
dtype0k

Adam/mul_3MulAdam/ReadVariableOp_2:value:0Adam/truediv:z:0*
T0*
_output_shapes

:�
Adam/Sqrt_1/ReadVariableOpReadVariableOp"adam_sub_3_readvariableop_resource^Adam/AssignAddVariableOp_1*
_output_shapes

:*
dtype0`
Adam/Sqrt_1Sqrt"Adam/Sqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Q
Adam/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3d

Adam/add_1AddV2Adam/Sqrt_1:y:0Adam/add_1/y:output:0*
T0*
_output_shapes

:b
Adam/truediv_1RealDivAdam/mul_3:z:0Adam/add_1:z:0*
T0*
_output_shapes

:�
Adam/AssignSubVariableOpAssignSubVariableOp*model_dense_matmul_readvariableop_resourceAdam/truediv_1:z:0"^model/dense/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0j
Adam/ReadVariableOp_3ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rj

Adam/add_2AddV2Adam/ReadVariableOp_3:value:0Adam/add_2/y:output:0*
T0	*
_output_shapes
: S
Adam/Cast_3CastAdam/add_2:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
Adam/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?[

Adam/Pow_2PowAdam/Cast_4/x:output:0Adam/Cast_3:y:0*
T0*
_output_shapes
: R
Adam/Cast_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?[

Adam/Pow_3PowAdam/Cast_5/x:output:0Adam/Cast_3:y:0*
T0*
_output_shapes
: Q
Adam/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_4SubAdam/sub_4/x:output:0Adam/Pow_3:z:0*
T0*
_output_shapes
: D
Adam/Sqrt_2SqrtAdam/sub_4:z:0*
T0*
_output_shapes
: l
Adam/ReadVariableOp_4ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0b

Adam/mul_4MulAdam/ReadVariableOp_4:value:0Adam/Sqrt_2:y:0*
T0*
_output_shapes
: Q
Adam/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_5SubAdam/sub_5/x:output:0Adam/Pow_2:z:0*
T0*
_output_shapes
: Z
Adam/truediv_2RealDivAdam/mul_4:z:0Adam/sub_5:z:0*
T0*
_output_shapes
: x
Adam/sub_6/ReadVariableOpReadVariableOp"adam_sub_6_readvariableop_resource*
_output_shapes
:*
dtype0m

Adam/sub_6SubIdentityN:output:1!Adam/sub_6/ReadVariableOp:value:0*
T0*
_output_shapes
:Q
Adam/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=]

Adam/mul_5MulAdam/sub_6:z:0Adam/mul_5/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_2AssignAddVariableOp"adam_sub_6_readvariableop_resourceAdam/mul_5:z:0^Adam/sub_6/ReadVariableOp*
_output_shapes
 *
dtype0P
Adam/Square_1SquareIdentityN:output:1*
T0*
_output_shapes
:x
Adam/sub_7/ReadVariableOpReadVariableOp"adam_sub_7_readvariableop_resource*
_output_shapes
:*
dtype0l

Adam/sub_7SubAdam/Square_1:y:0!Adam/sub_7/ReadVariableOp:value:0*
T0*
_output_shapes
:Q
Adam/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:]

Adam/mul_6MulAdam/sub_7:z:0Adam/mul_6/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_3AssignAddVariableOp"adam_sub_7_readvariableop_resourceAdam/mul_6:z:0^Adam/sub_7/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_5ReadVariableOp"adam_sub_6_readvariableop_resource^Adam/AssignAddVariableOp_2*
_output_shapes
:*
dtype0i

Adam/mul_7MulAdam/ReadVariableOp_5:value:0Adam/truediv_2:z:0*
T0*
_output_shapes
:�
Adam/Sqrt_3/ReadVariableOpReadVariableOp"adam_sub_7_readvariableop_resource^Adam/AssignAddVariableOp_3*
_output_shapes
:*
dtype0\
Adam/Sqrt_3Sqrt"Adam/Sqrt_3/ReadVariableOp:value:0*
T0*
_output_shapes
:Q
Adam/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3`

Adam/add_3AddV2Adam/Sqrt_3:y:0Adam/add_3/y:output:0*
T0*
_output_shapes
:^
Adam/truediv_3RealDivAdam/mul_7:z:0Adam/add_3:z:0*
T0*
_output_shapes
:�
Adam/AssignSubVariableOp_1AssignSubVariableOp+model_dense_biasadd_readvariableop_resourceAdam/truediv_3:z:0#^model/dense/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0j
Adam/ReadVariableOp_6ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_4/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rj

Adam/add_4AddV2Adam/ReadVariableOp_6:value:0Adam/add_4/y:output:0*
T0	*
_output_shapes
: S
Adam/Cast_6CastAdam/add_4:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
Adam/Cast_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?[

Adam/Pow_4PowAdam/Cast_7/x:output:0Adam/Cast_6:y:0*
T0*
_output_shapes
: R
Adam/Cast_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?[

Adam/Pow_5PowAdam/Cast_8/x:output:0Adam/Cast_6:y:0*
T0*
_output_shapes
: Q
Adam/sub_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_8SubAdam/sub_8/x:output:0Adam/Pow_5:z:0*
T0*
_output_shapes
: D
Adam/Sqrt_4SqrtAdam/sub_8:z:0*
T0*
_output_shapes
: l
Adam/ReadVariableOp_7ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0b

Adam/mul_8MulAdam/ReadVariableOp_7:value:0Adam/Sqrt_4:y:0*
T0*
_output_shapes
: Q
Adam/sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_9SubAdam/sub_9/x:output:0Adam/Pow_4:z:0*
T0*
_output_shapes
: Z
Adam/truediv_4RealDivAdam/mul_8:z:0Adam/sub_9:z:0*
T0*
_output_shapes
: 
Adam/sub_10/ReadVariableOpReadVariableOp#adam_sub_10_readvariableop_resource*
_output_shapes
:	�*
dtype0t
Adam/sub_10SubIdentityN:output:2"Adam/sub_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Q
Adam/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=c

Adam/mul_9MulAdam/sub_10:z:0Adam/mul_9/y:output:0*
T0*
_output_shapes
:	��
Adam/AssignAddVariableOp_4AssignAddVariableOp#adam_sub_10_readvariableop_resourceAdam/mul_9:z:0^Adam/sub_10/ReadVariableOp*
_output_shapes
 *
dtype0U
Adam/Square_2SquareIdentityN:output:2*
T0*
_output_shapes
:	�
Adam/sub_11/ReadVariableOpReadVariableOp#adam_sub_11_readvariableop_resource*
_output_shapes
:	�*
dtype0s
Adam/sub_11SubAdam/Square_2:y:0"Adam/sub_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�R
Adam/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:e
Adam/mul_10MulAdam/sub_11:z:0Adam/mul_10/y:output:0*
T0*
_output_shapes
:	��
Adam/AssignAddVariableOp_5AssignAddVariableOp#adam_sub_11_readvariableop_resourceAdam/mul_10:z:0^Adam/sub_11/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_8ReadVariableOp#adam_sub_10_readvariableop_resource^Adam/AssignAddVariableOp_4*
_output_shapes
:	�*
dtype0o
Adam/mul_11MulAdam/ReadVariableOp_8:value:0Adam/truediv_4:z:0*
T0*
_output_shapes
:	��
Adam/Sqrt_5/ReadVariableOpReadVariableOp#adam_sub_11_readvariableop_resource^Adam/AssignAddVariableOp_5*
_output_shapes
:	�*
dtype0a
Adam/Sqrt_5Sqrt"Adam/Sqrt_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Q
Adam/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e

Adam/add_5AddV2Adam/Sqrt_5:y:0Adam/add_5/y:output:0*
T0*
_output_shapes
:	�d
Adam/truediv_5RealDivAdam/mul_11:z:0Adam/add_5:z:0*
T0*
_output_shapes
:	��
Adam/AssignSubVariableOp_2AssignSubVariableOp,model_dense_1_matmul_readvariableop_resourceAdam/truediv_5:z:0$^model/dense_1/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0j
Adam/ReadVariableOp_9ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rj

Adam/add_6AddV2Adam/ReadVariableOp_9:value:0Adam/add_6/y:output:0*
T0	*
_output_shapes
: S
Adam/Cast_9CastAdam/add_6:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?\

Adam/Pow_6PowAdam/Cast_10/x:output:0Adam/Cast_9:y:0*
T0*
_output_shapes
: S
Adam/Cast_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?\

Adam/Pow_7PowAdam/Cast_11/x:output:0Adam/Cast_9:y:0*
T0*
_output_shapes
: R
Adam/sub_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_12SubAdam/sub_12/x:output:0Adam/Pow_7:z:0*
T0*
_output_shapes
: E
Adam/Sqrt_6SqrtAdam/sub_12:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_10ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0d
Adam/mul_12MulAdam/ReadVariableOp_10:value:0Adam/Sqrt_6:y:0*
T0*
_output_shapes
: R
Adam/sub_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_13SubAdam/sub_13/x:output:0Adam/Pow_6:z:0*
T0*
_output_shapes
: \
Adam/truediv_6RealDivAdam/mul_12:z:0Adam/sub_13:z:0*
T0*
_output_shapes
: {
Adam/sub_14/ReadVariableOpReadVariableOp#adam_sub_14_readvariableop_resource*
_output_shapes	
:�*
dtype0p
Adam/sub_14SubIdentityN:output:3"Adam/sub_14/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=a
Adam/mul_13MulAdam/sub_14:z:0Adam/mul_13/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_6AssignAddVariableOp#adam_sub_14_readvariableop_resourceAdam/mul_13:z:0^Adam/sub_14/ReadVariableOp*
_output_shapes
 *
dtype0Q
Adam/Square_3SquareIdentityN:output:3*
T0*
_output_shapes	
:�{
Adam/sub_15/ReadVariableOpReadVariableOp#adam_sub_15_readvariableop_resource*
_output_shapes	
:�*
dtype0o
Adam/sub_15SubAdam/Square_3:y:0"Adam/sub_15/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
Adam/mul_14MulAdam/sub_15:z:0Adam/mul_14/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_7AssignAddVariableOp#adam_sub_15_readvariableop_resourceAdam/mul_14:z:0^Adam/sub_15/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_11ReadVariableOp#adam_sub_14_readvariableop_resource^Adam/AssignAddVariableOp_6*
_output_shapes	
:�*
dtype0l
Adam/mul_15MulAdam/ReadVariableOp_11:value:0Adam/truediv_6:z:0*
T0*
_output_shapes	
:��
Adam/Sqrt_7/ReadVariableOpReadVariableOp#adam_sub_15_readvariableop_resource^Adam/AssignAddVariableOp_7*
_output_shapes	
:�*
dtype0]
Adam/Sqrt_7Sqrt"Adam/Sqrt_7/ReadVariableOp:value:0*
T0*
_output_shapes	
:�Q
Adam/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a

Adam/add_7AddV2Adam/Sqrt_7:y:0Adam/add_7/y:output:0*
T0*
_output_shapes	
:�`
Adam/truediv_7RealDivAdam/mul_15:z:0Adam/add_7:z:0*
T0*
_output_shapes	
:��
Adam/AssignSubVariableOp_3AssignSubVariableOp-model_dense_1_biasadd_readvariableop_resourceAdam/truediv_7:z:0%^model/dense_1/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_12ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rk

Adam/add_8AddV2Adam/ReadVariableOp_12:value:0Adam/add_8/y:output:0*
T0	*
_output_shapes
: T
Adam/Cast_12CastAdam/add_8:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]

Adam/Pow_8PowAdam/Cast_13/x:output:0Adam/Cast_12:y:0*
T0*
_output_shapes
: S
Adam/Cast_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?]

Adam/Pow_9PowAdam/Cast_14/x:output:0Adam/Cast_12:y:0*
T0*
_output_shapes
: R
Adam/sub_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_16SubAdam/sub_16/x:output:0Adam/Pow_9:z:0*
T0*
_output_shapes
: E
Adam/Sqrt_8SqrtAdam/sub_16:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_13ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0d
Adam/mul_16MulAdam/ReadVariableOp_13:value:0Adam/Sqrt_8:y:0*
T0*
_output_shapes
: R
Adam/sub_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_17SubAdam/sub_17/x:output:0Adam/Pow_8:z:0*
T0*
_output_shapes
: \
Adam/truediv_8RealDivAdam/mul_16:z:0Adam/sub_17:z:0*
T0*
_output_shapes
: �
Adam/sub_18/ReadVariableOpReadVariableOp#adam_sub_18_readvariableop_resource* 
_output_shapes
:
��*
dtype0u
Adam/sub_18SubIdentityN:output:4"Adam/sub_18/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��R
Adam/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=f
Adam/mul_17MulAdam/sub_18:z:0Adam/mul_17/y:output:0*
T0* 
_output_shapes
:
���
Adam/AssignAddVariableOp_8AssignAddVariableOp#adam_sub_18_readvariableop_resourceAdam/mul_17:z:0^Adam/sub_18/ReadVariableOp*
_output_shapes
 *
dtype0V
Adam/Square_4SquareIdentityN:output:4*
T0* 
_output_shapes
:
���
Adam/sub_19/ReadVariableOpReadVariableOp#adam_sub_19_readvariableop_resource* 
_output_shapes
:
��*
dtype0t
Adam/sub_19SubAdam/Square_4:y:0"Adam/sub_19/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��R
Adam/mul_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:f
Adam/mul_18MulAdam/sub_19:z:0Adam/mul_18/y:output:0*
T0* 
_output_shapes
:
���
Adam/AssignAddVariableOp_9AssignAddVariableOp#adam_sub_19_readvariableop_resourceAdam/mul_18:z:0^Adam/sub_19/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_14ReadVariableOp#adam_sub_18_readvariableop_resource^Adam/AssignAddVariableOp_8* 
_output_shapes
:
��*
dtype0q
Adam/mul_19MulAdam/ReadVariableOp_14:value:0Adam/truediv_8:z:0*
T0* 
_output_shapes
:
���
Adam/Sqrt_9/ReadVariableOpReadVariableOp#adam_sub_19_readvariableop_resource^Adam/AssignAddVariableOp_9* 
_output_shapes
:
��*
dtype0b
Adam/Sqrt_9Sqrt"Adam/Sqrt_9/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��Q
Adam/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3f

Adam/add_9AddV2Adam/Sqrt_9:y:0Adam/add_9/y:output:0*
T0* 
_output_shapes
:
��e
Adam/truediv_9RealDivAdam/mul_19:z:0Adam/add_9:z:0*
T0* 
_output_shapes
:
���
Adam/AssignSubVariableOp_4AssignSubVariableOp,model_dense_2_matmul_readvariableop_resourceAdam/truediv_9:z:0$^model/dense_2/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_15ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_10/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_10AddV2Adam/ReadVariableOp_15:value:0Adam/add_10/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_15CastAdam/add_10:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_10PowAdam/Cast_16/x:output:0Adam/Cast_15:y:0*
T0*
_output_shapes
: S
Adam/Cast_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_11PowAdam/Cast_17/x:output:0Adam/Cast_15:y:0*
T0*
_output_shapes
: R
Adam/sub_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_20SubAdam/sub_20/x:output:0Adam/Pow_11:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_10SqrtAdam/sub_20:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_16ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_20MulAdam/ReadVariableOp_16:value:0Adam/Sqrt_10:y:0*
T0*
_output_shapes
: R
Adam/sub_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_21SubAdam/sub_21/x:output:0Adam/Pow_10:z:0*
T0*
_output_shapes
: ]
Adam/truediv_10RealDivAdam/mul_20:z:0Adam/sub_21:z:0*
T0*
_output_shapes
: {
Adam/sub_22/ReadVariableOpReadVariableOp#adam_sub_22_readvariableop_resource*
_output_shapes	
:�*
dtype0p
Adam/sub_22SubIdentityN:output:5"Adam/sub_22/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=a
Adam/mul_21MulAdam/sub_22:z:0Adam/mul_21/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_10AssignAddVariableOp#adam_sub_22_readvariableop_resourceAdam/mul_21:z:0^Adam/sub_22/ReadVariableOp*
_output_shapes
 *
dtype0Q
Adam/Square_5SquareIdentityN:output:5*
T0*
_output_shapes	
:�{
Adam/sub_23/ReadVariableOpReadVariableOp#adam_sub_23_readvariableop_resource*
_output_shapes	
:�*
dtype0o
Adam/sub_23SubAdam/Square_5:y:0"Adam/sub_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
Adam/mul_22MulAdam/sub_23:z:0Adam/mul_22/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_11AssignAddVariableOp#adam_sub_23_readvariableop_resourceAdam/mul_22:z:0^Adam/sub_23/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_17ReadVariableOp#adam_sub_22_readvariableop_resource^Adam/AssignAddVariableOp_10*
_output_shapes	
:�*
dtype0m
Adam/mul_23MulAdam/ReadVariableOp_17:value:0Adam/truediv_10:z:0*
T0*
_output_shapes	
:��
Adam/Sqrt_11/ReadVariableOpReadVariableOp#adam_sub_23_readvariableop_resource^Adam/AssignAddVariableOp_11*
_output_shapes	
:�*
dtype0_
Adam/Sqrt_11Sqrt#Adam/Sqrt_11/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3d
Adam/add_11AddV2Adam/Sqrt_11:y:0Adam/add_11/y:output:0*
T0*
_output_shapes	
:�b
Adam/truediv_11RealDivAdam/mul_23:z:0Adam/add_11:z:0*
T0*
_output_shapes	
:��
Adam/AssignSubVariableOp_5AssignSubVariableOp-model_dense_2_biasadd_readvariableop_resourceAdam/truediv_11:z:0%^model/dense_2/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_18ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_12AddV2Adam/ReadVariableOp_18:value:0Adam/add_12/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_18CastAdam/add_12:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_12PowAdam/Cast_19/x:output:0Adam/Cast_18:y:0*
T0*
_output_shapes
: S
Adam/Cast_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_13PowAdam/Cast_20/x:output:0Adam/Cast_18:y:0*
T0*
_output_shapes
: R
Adam/sub_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_24SubAdam/sub_24/x:output:0Adam/Pow_13:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_12SqrtAdam/sub_24:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_19ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_24MulAdam/ReadVariableOp_19:value:0Adam/Sqrt_12:y:0*
T0*
_output_shapes
: R
Adam/sub_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_25SubAdam/sub_25/x:output:0Adam/Pow_12:z:0*
T0*
_output_shapes
: ]
Adam/truediv_12RealDivAdam/mul_24:z:0Adam/sub_25:z:0*
T0*
_output_shapes
: 
Adam/sub_26/ReadVariableOpReadVariableOp#adam_sub_26_readvariableop_resource*
_output_shapes
:	�@*
dtype0t
Adam/sub_26SubIdentityN:output:6"Adam/sub_26/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@R
Adam/mul_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=e
Adam/mul_25MulAdam/sub_26:z:0Adam/mul_25/y:output:0*
T0*
_output_shapes
:	�@�
Adam/AssignAddVariableOp_12AssignAddVariableOp#adam_sub_26_readvariableop_resourceAdam/mul_25:z:0^Adam/sub_26/ReadVariableOp*
_output_shapes
 *
dtype0U
Adam/Square_6SquareIdentityN:output:6*
T0*
_output_shapes
:	�@
Adam/sub_27/ReadVariableOpReadVariableOp#adam_sub_27_readvariableop_resource*
_output_shapes
:	�@*
dtype0s
Adam/sub_27SubAdam/Square_6:y:0"Adam/sub_27/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@R
Adam/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:e
Adam/mul_26MulAdam/sub_27:z:0Adam/mul_26/y:output:0*
T0*
_output_shapes
:	�@�
Adam/AssignAddVariableOp_13AssignAddVariableOp#adam_sub_27_readvariableop_resourceAdam/mul_26:z:0^Adam/sub_27/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_20ReadVariableOp#adam_sub_26_readvariableop_resource^Adam/AssignAddVariableOp_12*
_output_shapes
:	�@*
dtype0q
Adam/mul_27MulAdam/ReadVariableOp_20:value:0Adam/truediv_12:z:0*
T0*
_output_shapes
:	�@�
Adam/Sqrt_13/ReadVariableOpReadVariableOp#adam_sub_27_readvariableop_resource^Adam/AssignAddVariableOp_13*
_output_shapes
:	�@*
dtype0c
Adam/Sqrt_13Sqrt#Adam/Sqrt_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@R
Adam/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3h
Adam/add_13AddV2Adam/Sqrt_13:y:0Adam/add_13/y:output:0*
T0*
_output_shapes
:	�@f
Adam/truediv_13RealDivAdam/mul_27:z:0Adam/add_13:z:0*
T0*
_output_shapes
:	�@�
Adam/AssignSubVariableOp_6AssignSubVariableOp,model_dense_3_matmul_readvariableop_resourceAdam/truediv_13:z:0$^model/dense_3/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_21ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_14AddV2Adam/ReadVariableOp_21:value:0Adam/add_14/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_21CastAdam/add_14:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_14PowAdam/Cast_22/x:output:0Adam/Cast_21:y:0*
T0*
_output_shapes
: S
Adam/Cast_23/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_15PowAdam/Cast_23/x:output:0Adam/Cast_21:y:0*
T0*
_output_shapes
: R
Adam/sub_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_28SubAdam/sub_28/x:output:0Adam/Pow_15:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_14SqrtAdam/sub_28:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_22ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_28MulAdam/ReadVariableOp_22:value:0Adam/Sqrt_14:y:0*
T0*
_output_shapes
: R
Adam/sub_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_29SubAdam/sub_29/x:output:0Adam/Pow_14:z:0*
T0*
_output_shapes
: ]
Adam/truediv_14RealDivAdam/mul_28:z:0Adam/sub_29:z:0*
T0*
_output_shapes
: z
Adam/sub_30/ReadVariableOpReadVariableOp#adam_sub_30_readvariableop_resource*
_output_shapes
:@*
dtype0o
Adam/sub_30SubIdentityN:output:7"Adam/sub_30/ReadVariableOp:value:0*
T0*
_output_shapes
:@R
Adam/mul_29/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=`
Adam/mul_29MulAdam/sub_30:z:0Adam/mul_29/y:output:0*
T0*
_output_shapes
:@�
Adam/AssignAddVariableOp_14AssignAddVariableOp#adam_sub_30_readvariableop_resourceAdam/mul_29:z:0^Adam/sub_30/ReadVariableOp*
_output_shapes
 *
dtype0P
Adam/Square_7SquareIdentityN:output:7*
T0*
_output_shapes
:@z
Adam/sub_31/ReadVariableOpReadVariableOp#adam_sub_31_readvariableop_resource*
_output_shapes
:@*
dtype0n
Adam/sub_31SubAdam/Square_7:y:0"Adam/sub_31/ReadVariableOp:value:0*
T0*
_output_shapes
:@R
Adam/mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:`
Adam/mul_30MulAdam/sub_31:z:0Adam/mul_30/y:output:0*
T0*
_output_shapes
:@�
Adam/AssignAddVariableOp_15AssignAddVariableOp#adam_sub_31_readvariableop_resourceAdam/mul_30:z:0^Adam/sub_31/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_23ReadVariableOp#adam_sub_30_readvariableop_resource^Adam/AssignAddVariableOp_14*
_output_shapes
:@*
dtype0l
Adam/mul_31MulAdam/ReadVariableOp_23:value:0Adam/truediv_14:z:0*
T0*
_output_shapes
:@�
Adam/Sqrt_15/ReadVariableOpReadVariableOp#adam_sub_31_readvariableop_resource^Adam/AssignAddVariableOp_15*
_output_shapes
:@*
dtype0^
Adam/Sqrt_15Sqrt#Adam/Sqrt_15/ReadVariableOp:value:0*
T0*
_output_shapes
:@R
Adam/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3c
Adam/add_15AddV2Adam/Sqrt_15:y:0Adam/add_15/y:output:0*
T0*
_output_shapes
:@a
Adam/truediv_15RealDivAdam/mul_31:z:0Adam/add_15:z:0*
T0*
_output_shapes
:@�
Adam/AssignSubVariableOp_7AssignSubVariableOp-model_dense_3_biasadd_readvariableop_resourceAdam/truediv_15:z:0%^model/dense_3/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_24ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_16/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_16AddV2Adam/ReadVariableOp_24:value:0Adam/add_16/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_24CastAdam/add_16:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_16PowAdam/Cast_25/x:output:0Adam/Cast_24:y:0*
T0*
_output_shapes
: S
Adam/Cast_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_17PowAdam/Cast_26/x:output:0Adam/Cast_24:y:0*
T0*
_output_shapes
: R
Adam/sub_32/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_32SubAdam/sub_32/x:output:0Adam/Pow_17:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_16SqrtAdam/sub_32:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_25ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_32MulAdam/ReadVariableOp_25:value:0Adam/Sqrt_16:y:0*
T0*
_output_shapes
: R
Adam/sub_33/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_33SubAdam/sub_33/x:output:0Adam/Pow_16:z:0*
T0*
_output_shapes
: ]
Adam/truediv_16RealDivAdam/mul_32:z:0Adam/sub_33:z:0*
T0*
_output_shapes
: ~
Adam/sub_34/ReadVariableOpReadVariableOp#adam_sub_34_readvariableop_resource*
_output_shapes

:@*
dtype0s
Adam/sub_34SubIdentityN:output:8"Adam/sub_34/ReadVariableOp:value:0*
T0*
_output_shapes

:@R
Adam/mul_33/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=d
Adam/mul_33MulAdam/sub_34:z:0Adam/mul_33/y:output:0*
T0*
_output_shapes

:@�
Adam/AssignAddVariableOp_16AssignAddVariableOp#adam_sub_34_readvariableop_resourceAdam/mul_33:z:0^Adam/sub_34/ReadVariableOp*
_output_shapes
 *
dtype0T
Adam/Square_8SquareIdentityN:output:8*
T0*
_output_shapes

:@~
Adam/sub_35/ReadVariableOpReadVariableOp#adam_sub_35_readvariableop_resource*
_output_shapes

:@*
dtype0r
Adam/sub_35SubAdam/Square_8:y:0"Adam/sub_35/ReadVariableOp:value:0*
T0*
_output_shapes

:@R
Adam/mul_34/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:d
Adam/mul_34MulAdam/sub_35:z:0Adam/mul_34/y:output:0*
T0*
_output_shapes

:@�
Adam/AssignAddVariableOp_17AssignAddVariableOp#adam_sub_35_readvariableop_resourceAdam/mul_34:z:0^Adam/sub_35/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_26ReadVariableOp#adam_sub_34_readvariableop_resource^Adam/AssignAddVariableOp_16*
_output_shapes

:@*
dtype0p
Adam/mul_35MulAdam/ReadVariableOp_26:value:0Adam/truediv_16:z:0*
T0*
_output_shapes

:@�
Adam/Sqrt_17/ReadVariableOpReadVariableOp#adam_sub_35_readvariableop_resource^Adam/AssignAddVariableOp_17*
_output_shapes

:@*
dtype0b
Adam/Sqrt_17Sqrt#Adam/Sqrt_17/ReadVariableOp:value:0*
T0*
_output_shapes

:@R
Adam/add_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3g
Adam/add_17AddV2Adam/Sqrt_17:y:0Adam/add_17/y:output:0*
T0*
_output_shapes

:@e
Adam/truediv_17RealDivAdam/mul_35:z:0Adam/add_17:z:0*
T0*
_output_shapes

:@�
Adam/AssignSubVariableOp_8AssignSubVariableOp,model_dense_4_matmul_readvariableop_resourceAdam/truediv_17:z:0$^model/dense_4/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_27ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_18/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_18AddV2Adam/ReadVariableOp_27:value:0Adam/add_18/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_27CastAdam/add_18:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_18PowAdam/Cast_28/x:output:0Adam/Cast_27:y:0*
T0*
_output_shapes
: S
Adam/Cast_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_19PowAdam/Cast_29/x:output:0Adam/Cast_27:y:0*
T0*
_output_shapes
: R
Adam/sub_36/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_36SubAdam/sub_36/x:output:0Adam/Pow_19:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_18SqrtAdam/sub_36:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_28ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_36MulAdam/ReadVariableOp_28:value:0Adam/Sqrt_18:y:0*
T0*
_output_shapes
: R
Adam/sub_37/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_37SubAdam/sub_37/x:output:0Adam/Pow_18:z:0*
T0*
_output_shapes
: ]
Adam/truediv_18RealDivAdam/mul_36:z:0Adam/sub_37:z:0*
T0*
_output_shapes
: z
Adam/sub_38/ReadVariableOpReadVariableOp#adam_sub_38_readvariableop_resource*
_output_shapes
:*
dtype0o
Adam/sub_38SubIdentityN:output:9"Adam/sub_38/ReadVariableOp:value:0*
T0*
_output_shapes
:R
Adam/mul_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=`
Adam/mul_37MulAdam/sub_38:z:0Adam/mul_37/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_18AssignAddVariableOp#adam_sub_38_readvariableop_resourceAdam/mul_37:z:0^Adam/sub_38/ReadVariableOp*
_output_shapes
 *
dtype0P
Adam/Square_9SquareIdentityN:output:9*
T0*
_output_shapes
:z
Adam/sub_39/ReadVariableOpReadVariableOp#adam_sub_39_readvariableop_resource*
_output_shapes
:*
dtype0n
Adam/sub_39SubAdam/Square_9:y:0"Adam/sub_39/ReadVariableOp:value:0*
T0*
_output_shapes
:R
Adam/mul_38/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:`
Adam/mul_38MulAdam/sub_39:z:0Adam/mul_38/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_19AssignAddVariableOp#adam_sub_39_readvariableop_resourceAdam/mul_38:z:0^Adam/sub_39/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_29ReadVariableOp#adam_sub_38_readvariableop_resource^Adam/AssignAddVariableOp_18*
_output_shapes
:*
dtype0l
Adam/mul_39MulAdam/ReadVariableOp_29:value:0Adam/truediv_18:z:0*
T0*
_output_shapes
:�
Adam/Sqrt_19/ReadVariableOpReadVariableOp#adam_sub_39_readvariableop_resource^Adam/AssignAddVariableOp_19*
_output_shapes
:*
dtype0^
Adam/Sqrt_19Sqrt#Adam/Sqrt_19/ReadVariableOp:value:0*
T0*
_output_shapes
:R
Adam/add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3c
Adam/add_19AddV2Adam/Sqrt_19:y:0Adam/add_19/y:output:0*
T0*
_output_shapes
:a
Adam/truediv_19RealDivAdam/mul_39:z:0Adam/add_19:z:0*
T0*
_output_shapes
:�
Adam/AssignSubVariableOp_9AssignSubVariableOp-model_dense_4_biasadd_readvariableop_resourceAdam/truediv_19:z:0%^model/dense_4/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0L

Adam/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R�
Adam/AssignAddVariableOp_20AssignAddVariableOpadam_readvariableop_resourceAdam/Const:output:0^Adam/ReadVariableOp^Adam/ReadVariableOp_12^Adam/ReadVariableOp_15^Adam/ReadVariableOp_18^Adam/ReadVariableOp_21^Adam/ReadVariableOp_24^Adam/ReadVariableOp_27^Adam/ReadVariableOp_3^Adam/ReadVariableOp_6^Adam/ReadVariableOp_9*
_output_shapes
 *
dtype0	[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������\
ArgMaxArgMaxyArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Shape_1ShapeArgMax:output:0*
T0	*
_output_shapes
::��]
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������x
ArgMax_1ArgMaxmodel/dense_4/Sigmoid:y:0ArgMax_1/dimension:output:0*
T0*#
_output_shapes
:���������`
EqualEqualArgMax:output:0ArgMax_1:output:0*
T0	*#
_output_shapes
:���������V
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*#
_output_shapes
:���������O
ConstConst*
_output_shapes
:*
dtype0*
valueB: q
Sum_2Sum
Cast_1:y:0Const:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype09
SizeSize
Cast_1:y:0*
T0*
_output_shapes
: M
Cast_2CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_2:y:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: H
Identity_10Identitydiv_no_nan:z:0*
T0*
_output_shapes
: �
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype0�
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0�
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: J
Identity_11Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: U
Identity_12IdentityIdentity_11:output:0^NoOp*
T0*
_output_shapes
: U
Identity_13IdentityIdentity_10:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^Adam/AssignAddVariableOp^Adam/AssignAddVariableOp_1^Adam/AssignAddVariableOp_10^Adam/AssignAddVariableOp_11^Adam/AssignAddVariableOp_12^Adam/AssignAddVariableOp_13^Adam/AssignAddVariableOp_14^Adam/AssignAddVariableOp_15^Adam/AssignAddVariableOp_16^Adam/AssignAddVariableOp_17^Adam/AssignAddVariableOp_18^Adam/AssignAddVariableOp_19^Adam/AssignAddVariableOp_2^Adam/AssignAddVariableOp_20^Adam/AssignAddVariableOp_3^Adam/AssignAddVariableOp_4^Adam/AssignAddVariableOp_5^Adam/AssignAddVariableOp_6^Adam/AssignAddVariableOp_7^Adam/AssignAddVariableOp_8^Adam/AssignAddVariableOp_9^Adam/AssignSubVariableOp^Adam/AssignSubVariableOp_1^Adam/AssignSubVariableOp_2^Adam/AssignSubVariableOp_3^Adam/AssignSubVariableOp_4^Adam/AssignSubVariableOp_5^Adam/AssignSubVariableOp_6^Adam/AssignSubVariableOp_7^Adam/AssignSubVariableOp_8^Adam/AssignSubVariableOp_9^Adam/ReadVariableOp^Adam/ReadVariableOp_1^Adam/ReadVariableOp_10^Adam/ReadVariableOp_11^Adam/ReadVariableOp_12^Adam/ReadVariableOp_13^Adam/ReadVariableOp_14^Adam/ReadVariableOp_15^Adam/ReadVariableOp_16^Adam/ReadVariableOp_17^Adam/ReadVariableOp_18^Adam/ReadVariableOp_19^Adam/ReadVariableOp_2^Adam/ReadVariableOp_20^Adam/ReadVariableOp_21^Adam/ReadVariableOp_22^Adam/ReadVariableOp_23^Adam/ReadVariableOp_24^Adam/ReadVariableOp_25^Adam/ReadVariableOp_26^Adam/ReadVariableOp_27^Adam/ReadVariableOp_28^Adam/ReadVariableOp_29^Adam/ReadVariableOp_3^Adam/ReadVariableOp_4^Adam/ReadVariableOp_5^Adam/ReadVariableOp_6^Adam/ReadVariableOp_7^Adam/ReadVariableOp_8^Adam/ReadVariableOp_9^Adam/Sqrt_1/ReadVariableOp^Adam/Sqrt_11/ReadVariableOp^Adam/Sqrt_13/ReadVariableOp^Adam/Sqrt_15/ReadVariableOp^Adam/Sqrt_17/ReadVariableOp^Adam/Sqrt_19/ReadVariableOp^Adam/Sqrt_3/ReadVariableOp^Adam/Sqrt_5/ReadVariableOp^Adam/Sqrt_7/ReadVariableOp^Adam/Sqrt_9/ReadVariableOp^Adam/sub_10/ReadVariableOp^Adam/sub_11/ReadVariableOp^Adam/sub_14/ReadVariableOp^Adam/sub_15/ReadVariableOp^Adam/sub_18/ReadVariableOp^Adam/sub_19/ReadVariableOp^Adam/sub_2/ReadVariableOp^Adam/sub_22/ReadVariableOp^Adam/sub_23/ReadVariableOp^Adam/sub_26/ReadVariableOp^Adam/sub_27/ReadVariableOp^Adam/sub_3/ReadVariableOp^Adam/sub_30/ReadVariableOp^Adam/sub_31/ReadVariableOp^Adam/sub_34/ReadVariableOp^Adam/sub_35/ReadVariableOp^Adam/sub_38/ReadVariableOp^Adam/sub_39/ReadVariableOp^Adam/sub_6/ReadVariableOp^Adam/sub_7/ReadVariableOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
_output_shapes
 "#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
Adam/AssignAddVariableOp_10Adam/AssignAddVariableOp_102:
Adam/AssignAddVariableOp_11Adam/AssignAddVariableOp_112:
Adam/AssignAddVariableOp_12Adam/AssignAddVariableOp_122:
Adam/AssignAddVariableOp_13Adam/AssignAddVariableOp_132:
Adam/AssignAddVariableOp_14Adam/AssignAddVariableOp_142:
Adam/AssignAddVariableOp_15Adam/AssignAddVariableOp_152:
Adam/AssignAddVariableOp_16Adam/AssignAddVariableOp_162:
Adam/AssignAddVariableOp_17Adam/AssignAddVariableOp_172:
Adam/AssignAddVariableOp_18Adam/AssignAddVariableOp_182:
Adam/AssignAddVariableOp_19Adam/AssignAddVariableOp_1928
Adam/AssignAddVariableOp_1Adam/AssignAddVariableOp_12:
Adam/AssignAddVariableOp_20Adam/AssignAddVariableOp_2028
Adam/AssignAddVariableOp_2Adam/AssignAddVariableOp_228
Adam/AssignAddVariableOp_3Adam/AssignAddVariableOp_328
Adam/AssignAddVariableOp_4Adam/AssignAddVariableOp_428
Adam/AssignAddVariableOp_5Adam/AssignAddVariableOp_528
Adam/AssignAddVariableOp_6Adam/AssignAddVariableOp_628
Adam/AssignAddVariableOp_7Adam/AssignAddVariableOp_728
Adam/AssignAddVariableOp_8Adam/AssignAddVariableOp_828
Adam/AssignAddVariableOp_9Adam/AssignAddVariableOp_924
Adam/AssignAddVariableOpAdam/AssignAddVariableOp28
Adam/AssignSubVariableOp_1Adam/AssignSubVariableOp_128
Adam/AssignSubVariableOp_2Adam/AssignSubVariableOp_228
Adam/AssignSubVariableOp_3Adam/AssignSubVariableOp_328
Adam/AssignSubVariableOp_4Adam/AssignSubVariableOp_428
Adam/AssignSubVariableOp_5Adam/AssignSubVariableOp_528
Adam/AssignSubVariableOp_6Adam/AssignSubVariableOp_628
Adam/AssignSubVariableOp_7Adam/AssignSubVariableOp_728
Adam/AssignSubVariableOp_8Adam/AssignSubVariableOp_828
Adam/AssignSubVariableOp_9Adam/AssignSubVariableOp_924
Adam/AssignSubVariableOpAdam/AssignSubVariableOp20
Adam/ReadVariableOp_10Adam/ReadVariableOp_1020
Adam/ReadVariableOp_11Adam/ReadVariableOp_1120
Adam/ReadVariableOp_12Adam/ReadVariableOp_1220
Adam/ReadVariableOp_13Adam/ReadVariableOp_1320
Adam/ReadVariableOp_14Adam/ReadVariableOp_1420
Adam/ReadVariableOp_15Adam/ReadVariableOp_1520
Adam/ReadVariableOp_16Adam/ReadVariableOp_1620
Adam/ReadVariableOp_17Adam/ReadVariableOp_1720
Adam/ReadVariableOp_18Adam/ReadVariableOp_1820
Adam/ReadVariableOp_19Adam/ReadVariableOp_192.
Adam/ReadVariableOp_1Adam/ReadVariableOp_120
Adam/ReadVariableOp_20Adam/ReadVariableOp_2020
Adam/ReadVariableOp_21Adam/ReadVariableOp_2120
Adam/ReadVariableOp_22Adam/ReadVariableOp_2220
Adam/ReadVariableOp_23Adam/ReadVariableOp_2320
Adam/ReadVariableOp_24Adam/ReadVariableOp_2420
Adam/ReadVariableOp_25Adam/ReadVariableOp_2520
Adam/ReadVariableOp_26Adam/ReadVariableOp_2620
Adam/ReadVariableOp_27Adam/ReadVariableOp_2720
Adam/ReadVariableOp_28Adam/ReadVariableOp_2820
Adam/ReadVariableOp_29Adam/ReadVariableOp_292.
Adam/ReadVariableOp_2Adam/ReadVariableOp_22.
Adam/ReadVariableOp_3Adam/ReadVariableOp_32.
Adam/ReadVariableOp_4Adam/ReadVariableOp_42.
Adam/ReadVariableOp_5Adam/ReadVariableOp_52.
Adam/ReadVariableOp_6Adam/ReadVariableOp_62.
Adam/ReadVariableOp_7Adam/ReadVariableOp_72.
Adam/ReadVariableOp_8Adam/ReadVariableOp_82.
Adam/ReadVariableOp_9Adam/ReadVariableOp_92*
Adam/ReadVariableOpAdam/ReadVariableOp28
Adam/Sqrt_1/ReadVariableOpAdam/Sqrt_1/ReadVariableOp2:
Adam/Sqrt_11/ReadVariableOpAdam/Sqrt_11/ReadVariableOp2:
Adam/Sqrt_13/ReadVariableOpAdam/Sqrt_13/ReadVariableOp2:
Adam/Sqrt_15/ReadVariableOpAdam/Sqrt_15/ReadVariableOp2:
Adam/Sqrt_17/ReadVariableOpAdam/Sqrt_17/ReadVariableOp2:
Adam/Sqrt_19/ReadVariableOpAdam/Sqrt_19/ReadVariableOp28
Adam/Sqrt_3/ReadVariableOpAdam/Sqrt_3/ReadVariableOp28
Adam/Sqrt_5/ReadVariableOpAdam/Sqrt_5/ReadVariableOp28
Adam/Sqrt_7/ReadVariableOpAdam/Sqrt_7/ReadVariableOp28
Adam/Sqrt_9/ReadVariableOpAdam/Sqrt_9/ReadVariableOp28
Adam/sub_10/ReadVariableOpAdam/sub_10/ReadVariableOp28
Adam/sub_11/ReadVariableOpAdam/sub_11/ReadVariableOp28
Adam/sub_14/ReadVariableOpAdam/sub_14/ReadVariableOp28
Adam/sub_15/ReadVariableOpAdam/sub_15/ReadVariableOp28
Adam/sub_18/ReadVariableOpAdam/sub_18/ReadVariableOp28
Adam/sub_19/ReadVariableOpAdam/sub_19/ReadVariableOp26
Adam/sub_2/ReadVariableOpAdam/sub_2/ReadVariableOp28
Adam/sub_22/ReadVariableOpAdam/sub_22/ReadVariableOp28
Adam/sub_23/ReadVariableOpAdam/sub_23/ReadVariableOp28
Adam/sub_26/ReadVariableOpAdam/sub_26/ReadVariableOp28
Adam/sub_27/ReadVariableOpAdam/sub_27/ReadVariableOp26
Adam/sub_3/ReadVariableOpAdam/sub_3/ReadVariableOp28
Adam/sub_30/ReadVariableOpAdam/sub_30/ReadVariableOp28
Adam/sub_31/ReadVariableOpAdam/sub_31/ReadVariableOp28
Adam/sub_34/ReadVariableOpAdam/sub_34/ReadVariableOp28
Adam/sub_35/ReadVariableOpAdam/sub_35/ReadVariableOp28
Adam/sub_38/ReadVariableOpAdam/sub_38/ReadVariableOp28
Adam/sub_39/ReadVariableOpAdam/sub_39/ReadVariableOp26
Adam/sub_6/ReadVariableOpAdam/sub_6/ReadVariableOp26
Adam/sub_7/ReadVariableOpAdam/sub_7/ReadVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32*
AssignAddVariableOpAssignAddVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:JF
'
_output_shapes
:���������

_user_specified_namey:J F
'
_output_shapes
:���������

_user_specified_namex
�'
�
!__inference_internal_grad_fn_3498
result_grads_0
result_grads_1
result_grads_2
result_grads_3
result_grads_4
result_grads_5
result_grads_6
result_grads_7
result_grads_8
result_grads_9
result_grads_10
result_grads_11
result_grads_12
result_grads_13
result_grads_14
result_grads_15
result_grads_16
result_grads_17
result_grads_18
result_grads_19
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19M
IdentityIdentityresult_grads_0*
T0*
_output_shapes

:K

Identity_1Identityresult_grads_1*
T0*
_output_shapes
:P

Identity_2Identityresult_grads_2*
T0*
_output_shapes
:	�L

Identity_3Identityresult_grads_3*
T0*
_output_shapes	
:�Q

Identity_4Identityresult_grads_4*
T0* 
_output_shapes
:
��L

Identity_5Identityresult_grads_5*
T0*
_output_shapes	
:�P

Identity_6Identityresult_grads_6*
T0*
_output_shapes
:	�@K

Identity_7Identityresult_grads_7*
T0*
_output_shapes
:@O

Identity_8Identityresult_grads_8*
T0*
_output_shapes

:@K

Identity_9Identityresult_grads_9*
T0*
_output_shapes
:�
	IdentityN	IdentityNresult_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9*
T
2**
_gradient_op_typeCustomGradient-3457*�
_output_shapes�
�:::	�:�:
��:�:	�@:@:@::::	�:�:
��:�:	�@:@:@:T
Identity_10IdentityIdentityN:output:0*
T0*
_output_shapes

:P
Identity_11IdentityIdentityN:output:1*
T0*
_output_shapes
:U
Identity_12IdentityIdentityN:output:2*
T0*
_output_shapes
:	�Q
Identity_13IdentityIdentityN:output:3*
T0*
_output_shapes	
:�V
Identity_14IdentityIdentityN:output:4*
T0* 
_output_shapes
:
��Q
Identity_15IdentityIdentityN:output:5*
T0*
_output_shapes	
:�U
Identity_16IdentityIdentityN:output:6*
T0*
_output_shapes
:	�@P
Identity_17IdentityIdentityN:output:7*
T0*
_output_shapes
:@T
Identity_18IdentityIdentityN:output:8*
T0*
_output_shapes

:@P
Identity_19IdentityIdentityN:output:9*
T0*
_output_shapes
:"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:::	�:�:
��:�:	�@:@:@::::	�:�:
��:�:	�@:@:@::KG

_output_shapes
:
)
_user_specified_nameresult_grads_19:OK

_output_shapes

:@
)
_user_specified_nameresult_grads_18:KG

_output_shapes
:@
)
_user_specified_nameresult_grads_17:PL

_output_shapes
:	�@
)
_user_specified_nameresult_grads_16:LH

_output_shapes	
:�
)
_user_specified_nameresult_grads_15:QM
 
_output_shapes
:
��
)
_user_specified_nameresult_grads_14:LH

_output_shapes	
:�
)
_user_specified_nameresult_grads_13:PL

_output_shapes
:	�
)
_user_specified_nameresult_grads_12:KG

_output_shapes
:
)
_user_specified_nameresult_grads_11:O
K

_output_shapes

:
)
_user_specified_nameresult_grads_10:J	F

_output_shapes
:
(
_user_specified_nameresult_grads_9:NJ

_output_shapes

:@
(
_user_specified_nameresult_grads_8:JF

_output_shapes
:@
(
_user_specified_nameresult_grads_7:OK

_output_shapes
:	�@
(
_user_specified_nameresult_grads_6:KG

_output_shapes	
:�
(
_user_specified_nameresult_grads_5:PL
 
_output_shapes
:
��
(
_user_specified_nameresult_grads_4:KG

_output_shapes	
:�
(
_user_specified_nameresult_grads_3:OK

_output_shapes
:	�
(
_user_specified_nameresult_grads_2:JF

_output_shapes
:
(
_user_specified_nameresult_grads_1:N J

_output_shapes

:
(
_user_specified_nameresult_grads_0
�

�
?__inference_dense_layer_call_and_return_conditional_losses_3165

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_2923

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_3212

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_1927
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *j
_output_shapesX
V:::	�:�:
��:�:	�@:@:@:*,
_read_only_resource_inputs

 	*-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_parameters_236f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:i

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
:	�e

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*
_output_shapes	
:�j

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0* 
_output_shapes
:
��e

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*
_output_shapes	
:�i

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*
_output_shapes
:	�@d

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*
_output_shapes
:@h

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*
_output_shapes

:@d

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$	 

_user_specified_name1905:$ 

_user_specified_name1903:$ 

_user_specified_name1901:$ 

_user_specified_name1899:$ 

_user_specified_name1897:$ 

_user_specified_name1895:$ 

_user_specified_name1893:$ 

_user_specified_name1891:$ 

_user_specified_name1889:$  

_user_specified_name1887
�

�
A__inference_dense_2_layer_call_and_return_conditional_losses_2906

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_2952

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
D
(__inference_dropout_2_layer_call_fn_3289

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_3124`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_3301

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_3259

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�2
�
__inference_restore_298
a0
a1
a2
a3
a4
a5
a6
a7
a8
a9+
assignvariableop_resource:)
assignvariableop_1_resource:.
assignvariableop_2_resource:	�*
assignvariableop_3_resource:	�/
assignvariableop_4_resource:
��*
assignvariableop_5_resource:	�.
assignvariableop_6_resource:	�@)
assignvariableop_7_resource:@-
assignvariableop_8_resource:@)
assignvariableop_9_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�StatefulPartitionedCall|
AssignVariableOpAssignVariableOpassignvariableop_resourcea0*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourcea1*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_2AssignVariableOpassignvariableop_2_resourcea2*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_3AssignVariableOpassignvariableop_3_resourcea3*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_4AssignVariableOpassignvariableop_4_resourcea4*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_5AssignVariableOpassignvariableop_5_resourcea5*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_6AssignVariableOpassignvariableop_6_resourcea6*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_7AssignVariableOpassignvariableop_7_resourcea7*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_8AssignVariableOpassignvariableop_8_resourcea8*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_9AssignVariableOpassignvariableop_9_resourcea9*
_output_shapes
 *
dtype0*
validate_shape(�
StatefulPartitionedCallStatefulPartitionedCallassignvariableop_resourceassignvariableop_1_resourceassignvariableop_2_resourceassignvariableop_3_resourceassignvariableop_4_resourceassignvariableop_5_resourceassignvariableop_6_resourceassignvariableop_7_resourceassignvariableop_8_resourceassignvariableop_9_resource^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *j
_output_shapesX
V:::	�:�:
��:�:	�@:@:@:*,
_read_only_resource_inputs

 	*-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_parameters_236f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
:i

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
:	�e

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*
_output_shapes	
:�j

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0* 
_output_shapes
:
��e

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*
_output_shapes	
:�i

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*
_output_shapes
:	�@d

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*
_output_shapes
:@h

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*
_output_shapes

:@d

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:::	�:�:
��:�:	�@:@:@:: : : : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:>	:

_output_shapes
:

_user_specified_namea9:B>

_output_shapes

:@

_user_specified_namea8:>:

_output_shapes
:@

_user_specified_namea7:C?

_output_shapes
:	�@

_user_specified_namea6:?;

_output_shapes	
:�

_user_specified_namea5:D@
 
_output_shapes
:
��

_user_specified_namea4:?;

_output_shapes	
:�

_user_specified_namea3:C?

_output_shapes
:	�

_user_specified_namea2:>:

_output_shapes
:

_user_specified_namea1:B >

_output_shapes

:

_user_specified_namea0
�
_
&__inference_dropout_layer_call_fn_3190

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2894p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_2_layer_call_and_return_conditional_losses_3232

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�O
�
__inference_infer_2043
x<
*model_dense_matmul_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:?
,model_dense_1_matmul_readvariableop_resource:	�<
-model_dense_1_biasadd_readvariableop_resource:	�@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�?
,model_dense_3_matmul_readvariableop_resource:	�@;
-model_dense_3_biasadd_readvariableop_resource:@>
,model_dense_4_matmul_readvariableop_resource:@;
-model_dense_4_biasadd_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0|
model/dense/MatMulMatMulx)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������`
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout/dropout/MulMul model/dense_1/Relu:activations:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������y
model/dropout/dropout/ShapeShape model/dense_1/Relu:activations:0*
T0*
_output_shapes
::���
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0i
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������b
model/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
model/dropout/dropout/SelectV2SelectV2&model/dropout/dropout/GreaterEqual:z:0model/dropout/dropout/Mul:z:0&model/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul'model/dropout/dropout/SelectV2:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������b
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_1/dropout/MulMul model/dense_2/Relu:activations:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������{
model/dropout_1/dropout/ShapeShape model/dense_2/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0k
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������d
model/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_1/dropout/SelectV2SelectV2(model/dropout_1/dropout/GreaterEqual:z:0model/dropout_1/dropout/Mul:z:0(model/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_3/MatMulMatMul)model/dropout_1/dropout/SelectV2:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@b
model/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_2/dropout/MulMul model/dense_3/Relu:activations:0&model/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@{
model/dropout_2/dropout/ShapeShape model/dense_3/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_2/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0k
&model/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_2/dropout/GreaterEqualGreaterEqual=model/dropout_2/dropout/random_uniform/RandomUniform:output:0/model/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@d
model/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_2/dropout/SelectV2SelectV2(model/dropout_2/dropout/GreaterEqual:z:0model/dropout_2/dropout/Mul:z:0(model/dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_4/MatMulMatMul)model/dropout_2/dropout/SelectV2:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_4/SigmoidSigmoidmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitymodel/dense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
'
_output_shapes
:���������

_user_specified_namex
��
� 
__inference__traced_save_3724
file_prefix5
#read_disablecopyonread_dense_kernel:1
#read_1_disablecopyonread_dense_bias::
'read_2_disablecopyonread_dense_1_kernel:	�4
%read_3_disablecopyonread_dense_1_bias:	�;
'read_4_disablecopyonread_dense_2_kernel:
��4
%read_5_disablecopyonread_dense_2_bias:	�:
'read_6_disablecopyonread_dense_3_kernel:	�@3
%read_7_disablecopyonread_dense_3_bias:@9
'read_8_disablecopyonread_dense_4_kernel:@3
%read_9_disablecopyonread_dense_4_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: ?
-read_12_disablecopyonread_adam_m_dense_kernel:?
-read_13_disablecopyonread_adam_v_dense_kernel:9
+read_14_disablecopyonread_adam_m_dense_bias:9
+read_15_disablecopyonread_adam_v_dense_bias:B
/read_16_disablecopyonread_adam_m_dense_1_kernel:	�B
/read_17_disablecopyonread_adam_v_dense_1_kernel:	�<
-read_18_disablecopyonread_adam_m_dense_1_bias:	�<
-read_19_disablecopyonread_adam_v_dense_1_bias:	�C
/read_20_disablecopyonread_adam_m_dense_2_kernel:
��C
/read_21_disablecopyonread_adam_v_dense_2_kernel:
��<
-read_22_disablecopyonread_adam_m_dense_2_bias:	�<
-read_23_disablecopyonread_adam_v_dense_2_bias:	�B
/read_24_disablecopyonread_adam_m_dense_3_kernel:	�@B
/read_25_disablecopyonread_adam_v_dense_3_kernel:	�@;
-read_26_disablecopyonread_adam_m_dense_3_bias:@;
-read_27_disablecopyonread_adam_v_dense_3_bias:@A
/read_28_disablecopyonread_adam_m_dense_4_kernel:@A
/read_29_disablecopyonread_adam_v_dense_4_kernel:@;
-read_30_disablecopyonread_adam_m_dense_4_bias:;
-read_31_disablecopyonread_adam_v_dense_4_bias:+
!read_32_disablecopyonread_total_1: +
!read_33_disablecopyonread_count_1: )
read_34_disablecopyonread_total: )
read_35_disablecopyonread_count: 
savev2_const
identity_73��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: f
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead*
_output_shapes

:*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:h
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:l
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead*
_output_shapes
:	�*
dtype0_

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�j
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0`

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_3_kernel*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_3_kernel^Read_6/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0`
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@j
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_3_bias*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_3_bias^Read_7/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@l
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_4_kernel*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_4_kernel^Read_8/DisableCopyOnRead*
_output_shapes

:@*
dtype0_
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@j
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_4_bias*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_4_bias^Read_9/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:i
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead*
_output_shapes
: *
dtype0	X
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: m
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: s
Read_12/DisableCopyOnReadDisableCopyOnRead-read_12_disablecopyonread_adam_m_dense_kernel*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp-read_12_disablecopyonread_adam_m_dense_kernel^Read_12/DisableCopyOnRead*
_output_shapes

:*
dtype0`
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:s
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_adam_v_dense_kernel*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_adam_v_dense_kernel^Read_13/DisableCopyOnRead*
_output_shapes

:*
dtype0`
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:q
Read_14/DisableCopyOnReadDisableCopyOnRead+read_14_disablecopyonread_adam_m_dense_bias*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp+read_14_disablecopyonread_adam_m_dense_bias^Read_14/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:q
Read_15/DisableCopyOnReadDisableCopyOnRead+read_15_disablecopyonread_adam_v_dense_bias*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp+read_15_disablecopyonread_adam_v_dense_bias^Read_15/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:u
Read_16/DisableCopyOnReadDisableCopyOnRead/read_16_disablecopyonread_adam_m_dense_1_kernel*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp/read_16_disablecopyonread_adam_m_dense_1_kernel^Read_16/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	�u
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_adam_v_dense_1_kernel*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_adam_v_dense_1_kernel^Read_17/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	�s
Read_18/DisableCopyOnReadDisableCopyOnRead-read_18_disablecopyonread_adam_m_dense_1_bias*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp-read_18_disablecopyonread_adam_m_dense_1_bias^Read_18/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:�s
Read_19/DisableCopyOnReadDisableCopyOnRead-read_19_disablecopyonread_adam_v_dense_1_bias*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp-read_19_disablecopyonread_adam_v_dense_1_bias^Read_19/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�u
Read_20/DisableCopyOnReadDisableCopyOnRead/read_20_disablecopyonread_adam_m_dense_2_kernel*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp/read_20_disablecopyonread_adam_m_dense_2_kernel^Read_20/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��u
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_adam_v_dense_2_kernel*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_adam_v_dense_2_kernel^Read_21/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��s
Read_22/DisableCopyOnReadDisableCopyOnRead-read_22_disablecopyonread_adam_m_dense_2_bias*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp-read_22_disablecopyonread_adam_m_dense_2_bias^Read_22/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:�s
Read_23/DisableCopyOnReadDisableCopyOnRead-read_23_disablecopyonread_adam_v_dense_2_bias*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp-read_23_disablecopyonread_adam_v_dense_2_bias^Read_23/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�u
Read_24/DisableCopyOnReadDisableCopyOnRead/read_24_disablecopyonread_adam_m_dense_3_kernel*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp/read_24_disablecopyonread_adam_m_dense_3_kernel^Read_24/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0a
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@u
Read_25/DisableCopyOnReadDisableCopyOnRead/read_25_disablecopyonread_adam_v_dense_3_kernel*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp/read_25_disablecopyonread_adam_v_dense_3_kernel^Read_25/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0a
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@s
Read_26/DisableCopyOnReadDisableCopyOnRead-read_26_disablecopyonread_adam_m_dense_3_bias*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp-read_26_disablecopyonread_adam_m_dense_3_bias^Read_26/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@s
Read_27/DisableCopyOnReadDisableCopyOnRead-read_27_disablecopyonread_adam_v_dense_3_bias*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp-read_27_disablecopyonread_adam_v_dense_3_bias^Read_27/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@u
Read_28/DisableCopyOnReadDisableCopyOnRead/read_28_disablecopyonread_adam_m_dense_4_kernel*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp/read_28_disablecopyonread_adam_m_dense_4_kernel^Read_28/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:@u
Read_29/DisableCopyOnReadDisableCopyOnRead/read_29_disablecopyonread_adam_v_dense_4_kernel*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp/read_29_disablecopyonread_adam_v_dense_4_kernel^Read_29/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:@s
Read_30/DisableCopyOnReadDisableCopyOnRead-read_30_disablecopyonread_adam_m_dense_4_bias*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp-read_30_disablecopyonread_adam_m_dense_4_bias^Read_30/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:s
Read_31/DisableCopyOnReadDisableCopyOnRead-read_31_disablecopyonread_adam_v_dense_4_bias*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp-read_31_disablecopyonread_adam_v_dense_4_bias^Read_31/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:g
Read_32/DisableCopyOnReadDisableCopyOnRead!read_32_disablecopyonread_total_1*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp!read_32_disablecopyonread_total_1^Read_32/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: g
Read_33/DisableCopyOnReadDisableCopyOnRead!read_33_disablecopyonread_count_1*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp!read_33_disablecopyonread_count_1^Read_33/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: e
Read_34/DisableCopyOnReadDisableCopyOnReadread_34_disablecopyonread_total*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpread_34_disablecopyonread_total^Read_34/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: e
Read_35/DisableCopyOnReadDisableCopyOnReadread_35_disablecopyonread_count*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpread_35_disablecopyonread_count^Read_35/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6model/optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB9model/optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *3
dtypes)
'2%	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_72Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_73IdentityIdentity_72:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_73Identity_73:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=%9

_output_shapes
: 

_user_specified_nameConst:%$!

_user_specified_namecount:%#!

_user_specified_nametotal:'"#
!
_user_specified_name	count_1:'!#
!
_user_specified_name	total_1:3 /
-
_user_specified_nameAdam/v/dense_4/bias:3/
-
_user_specified_nameAdam/m/dense_4/bias:51
/
_user_specified_nameAdam/v/dense_4/kernel:51
/
_user_specified_nameAdam/m/dense_4/kernel:3/
-
_user_specified_nameAdam/v/dense_3/bias:3/
-
_user_specified_nameAdam/m/dense_3/bias:51
/
_user_specified_nameAdam/v/dense_3/kernel:51
/
_user_specified_nameAdam/m/dense_3/kernel:3/
-
_user_specified_nameAdam/v/dense_2/bias:3/
-
_user_specified_nameAdam/m/dense_2/bias:51
/
_user_specified_nameAdam/v/dense_2/kernel:51
/
_user_specified_nameAdam/m/dense_2/kernel:3/
-
_user_specified_nameAdam/v/dense_1/bias:3/
-
_user_specified_nameAdam/m/dense_1/bias:51
/
_user_specified_nameAdam/v/dense_1/kernel:51
/
_user_specified_nameAdam/m/dense_1/kernel:1-
+
_user_specified_nameAdam/v/dense/bias:1-
+
_user_specified_nameAdam/m/dense/bias:3/
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,
(
&
_user_specified_namedense_4/bias:.	*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�O
�
__inference_infer_1777
x<
*model_dense_matmul_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:?
,model_dense_1_matmul_readvariableop_resource:	�<
-model_dense_1_biasadd_readvariableop_resource:	�@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�?
,model_dense_3_matmul_readvariableop_resource:	�@;
-model_dense_3_biasadd_readvariableop_resource:@>
,model_dense_4_matmul_readvariableop_resource:@;
-model_dense_4_biasadd_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0|
model/dense/MatMulMatMulx)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������`
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout/dropout/MulMul model/dense_1/Relu:activations:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������y
model/dropout/dropout/ShapeShape model/dense_1/Relu:activations:0*
T0*
_output_shapes
::���
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0i
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������b
model/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
model/dropout/dropout/SelectV2SelectV2&model/dropout/dropout/GreaterEqual:z:0model/dropout/dropout/Mul:z:0&model/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul'model/dropout/dropout/SelectV2:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������b
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_1/dropout/MulMul model/dense_2/Relu:activations:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������{
model/dropout_1/dropout/ShapeShape model/dense_2/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0k
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������d
model/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_1/dropout/SelectV2SelectV2(model/dropout_1/dropout/GreaterEqual:z:0model/dropout_1/dropout/Mul:z:0(model/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_3/MatMulMatMul)model/dropout_1/dropout/SelectV2:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@b
model/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_2/dropout/MulMul model/dense_3/Relu:activations:0&model/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@{
model/dropout_2/dropout/ShapeShape model/dense_3/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_2/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0k
&model/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_2/dropout/GreaterEqualGreaterEqual=model/dropout_2/dropout/random_uniform/RandomUniform:output:0/model/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@d
model/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_2/dropout/SelectV2SelectV2(model/dropout_2/dropout/GreaterEqual:z:0model/dropout_2/dropout/Mul:z:0(model/dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_4/MatMulMatMul)model/dropout_2/dropout/SelectV2:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_4/SigmoidSigmoidmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitymodel/dense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
$__inference_model_layer_call_fn_3053
input_1
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_3003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$
 

_user_specified_name3049:$	 

_user_specified_name3047:$ 

_user_specified_name3045:$ 

_user_specified_name3043:$ 

_user_specified_name3041:$ 

_user_specified_name3039:$ 

_user_specified_name3037:$ 

_user_specified_name3035:$ 

_user_specified_name3033:$ 

_user_specified_name3031:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
&__inference_dense_2_layer_call_fn_3221

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2906p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name3217:$ 

_user_specified_name3215:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_4_layer_call_and_return_conditional_losses_2964

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_2894

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
__inference_parameters_2085.
read_readvariableop_resource:,
read_1_readvariableop_resource:1
read_2_readvariableop_resource:	�-
read_3_readvariableop_resource:	�2
read_4_readvariableop_resource:
��-
read_5_readvariableop_resource:	�1
read_6_readvariableop_resource:	�@,
read_7_readvariableop_resource:@0
read_8_readvariableop_resource:@,
read_9_readvariableop_resource:
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOp�Read_4/ReadVariableOp�Read_5/ReadVariableOp�Read_6/ReadVariableOp�Read_7/ReadVariableOp�Read_8/ReadVariableOp�Read_9/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:p
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:*
dtype0Z

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�q
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes	
:�*
dtype0[

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�v
Read_4/ReadVariableOpReadVariableOpread_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0`

Identity_4IdentityRead_4/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��q
Read_5/ReadVariableOpReadVariableOpread_5_readvariableop_resource*
_output_shapes	
:�*
dtype0[

Identity_5IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes	
:�u
Read_6/ReadVariableOpReadVariableOpread_6_readvariableop_resource*
_output_shapes
:	�@*
dtype0_

Identity_6IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@p
Read_7/ReadVariableOpReadVariableOpread_7_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_7IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@t
Read_8/ReadVariableOpReadVariableOpread_8_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_8IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_9/ReadVariableOpReadVariableOpread_9_readvariableop_resource*
_output_shapes
:*
dtype0Z

Identity_9IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:Z
Identity_10IdentityIdentity:output:0^NoOp*
T0*
_output_shapes

:X
Identity_11IdentityIdentity_1:output:0^NoOp*
T0*
_output_shapes
:]
Identity_12IdentityIdentity_2:output:0^NoOp*
T0*
_output_shapes
:	�Y
Identity_13IdentityIdentity_3:output:0^NoOp*
T0*
_output_shapes	
:�^
Identity_14IdentityIdentity_4:output:0^NoOp*
T0* 
_output_shapes
:
��Y
Identity_15IdentityIdentity_5:output:0^NoOp*
T0*
_output_shapes	
:�]
Identity_16IdentityIdentity_6:output:0^NoOp*
T0*
_output_shapes
:	�@X
Identity_17IdentityIdentity_7:output:0^NoOp*
T0*
_output_shapes
:@\
Identity_18IdentityIdentity_8:output:0^NoOp*
T0*
_output_shapes

:@X
Identity_19IdentityIdentity_9:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp^Read_4/ReadVariableOp^Read_5/ReadVariableOp^Read_6/ReadVariableOp^Read_7/ReadVariableOp^Read_8/ReadVariableOp^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp2.
Read_4/ReadVariableOpRead_4/ReadVariableOp2.
Read_5/ReadVariableOpRead_5/ReadVariableOp2.
Read_6/ReadVariableOpRead_6/ReadVariableOp2.
Read_7/ReadVariableOpRead_7/ReadVariableOp2.
Read_8/ReadVariableOpRead_8/ReadVariableOp2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource
�O
�
__inference__wrapped_model_2848
input_1<
*model_dense_matmul_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:?
,model_dense_1_matmul_readvariableop_resource:	�<
-model_dense_1_biasadd_readvariableop_resource:	�@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�?
,model_dense_3_matmul_readvariableop_resource:	�@;
-model_dense_3_biasadd_readvariableop_resource:@>
,model_dense_4_matmul_readvariableop_resource:@;
-model_dense_4_biasadd_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������`
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout/dropout/MulMul model/dense_1/Relu:activations:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������y
model/dropout/dropout/ShapeShape model/dense_1/Relu:activations:0*
T0*
_output_shapes
::���
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0i
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������b
model/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
model/dropout/dropout/SelectV2SelectV2&model/dropout/dropout/GreaterEqual:z:0model/dropout/dropout/Mul:z:0&model/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul'model/dropout/dropout/SelectV2:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������b
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_1/dropout/MulMul model/dense_2/Relu:activations:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������{
model/dropout_1/dropout/ShapeShape model/dense_2/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0k
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������d
model/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_1/dropout/SelectV2SelectV2(model/dropout_1/dropout/GreaterEqual:z:0model/dropout_1/dropout/Mul:z:0(model/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_3/MatMulMatMul)model/dropout_1/dropout/SelectV2:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@b
model/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_2/dropout/MulMul model/dense_3/Relu:activations:0&model/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@{
model/dropout_2/dropout/ShapeShape model/dense_3/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_2/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0k
&model/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_2/dropout/GreaterEqualGreaterEqual=model/dropout_2/dropout/random_uniform/RandomUniform:output:0/model/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@d
model/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_2/dropout/SelectV2SelectV2(model/dropout_2/dropout/GreaterEqual:z:0model/dropout_2/dropout/Mul:z:0(model/dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_4/MatMulMatMul)model/dropout_2/dropout/SelectV2:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_4/SigmoidSigmoidmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitymodel/dense_4/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
"__inference_signature_wrapper_1884
x
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *
fR
__inference_infer_1777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$
 

_user_specified_name1880:$	 

_user_specified_name1878:$ 

_user_specified_name1876:$ 

_user_specified_name1874:$ 

_user_specified_name1872:$ 

_user_specified_name1870:$ 

_user_specified_name1868:$ 

_user_specified_name1866:$ 

_user_specified_name1864:$ 

_user_specified_name1862:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
&__inference_dense_3_layer_call_fn_3268

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2935o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name3264:$ 

_user_specified_name3262:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_3124

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
A__inference_dense_4_layer_call_and_return_conditional_losses_3326

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_dense_1_layer_call_fn_3174

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2877p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name3170:$ 

_user_specified_name3168:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_3207

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�(
__inference_train_1714
x
y<
*model_dense_matmul_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:?
,model_dense_1_matmul_readvariableop_resource:	�<
-model_dense_1_biasadd_readvariableop_resource:	�@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�?
,model_dense_3_matmul_readvariableop_resource:	�@;
-model_dense_3_biasadd_readvariableop_resource:@>
,model_dense_4_matmul_readvariableop_resource:@;
-model_dense_4_biasadd_readvariableop_resource:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: &
adam_readvariableop_resource:	 (
adam_readvariableop_1_resource: 4
"adam_sub_2_readvariableop_resource:4
"adam_sub_3_readvariableop_resource:0
"adam_sub_6_readvariableop_resource:0
"adam_sub_7_readvariableop_resource:6
#adam_sub_10_readvariableop_resource:	�6
#adam_sub_11_readvariableop_resource:	�2
#adam_sub_14_readvariableop_resource:	�2
#adam_sub_15_readvariableop_resource:	�7
#adam_sub_18_readvariableop_resource:
��7
#adam_sub_19_readvariableop_resource:
��2
#adam_sub_22_readvariableop_resource:	�2
#adam_sub_23_readvariableop_resource:	�6
#adam_sub_26_readvariableop_resource:	�@6
#adam_sub_27_readvariableop_resource:	�@1
#adam_sub_30_readvariableop_resource:@1
#adam_sub_31_readvariableop_resource:@5
#adam_sub_34_readvariableop_resource:@5
#adam_sub_35_readvariableop_resource:@1
#adam_sub_38_readvariableop_resource:1
#adam_sub_39_readvariableop_resource:(
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: 
identity_12
identity_13��Adam/AssignAddVariableOp�Adam/AssignAddVariableOp_1�Adam/AssignAddVariableOp_10�Adam/AssignAddVariableOp_11�Adam/AssignAddVariableOp_12�Adam/AssignAddVariableOp_13�Adam/AssignAddVariableOp_14�Adam/AssignAddVariableOp_15�Adam/AssignAddVariableOp_16�Adam/AssignAddVariableOp_17�Adam/AssignAddVariableOp_18�Adam/AssignAddVariableOp_19�Adam/AssignAddVariableOp_2�Adam/AssignAddVariableOp_20�Adam/AssignAddVariableOp_3�Adam/AssignAddVariableOp_4�Adam/AssignAddVariableOp_5�Adam/AssignAddVariableOp_6�Adam/AssignAddVariableOp_7�Adam/AssignAddVariableOp_8�Adam/AssignAddVariableOp_9�Adam/AssignSubVariableOp�Adam/AssignSubVariableOp_1�Adam/AssignSubVariableOp_2�Adam/AssignSubVariableOp_3�Adam/AssignSubVariableOp_4�Adam/AssignSubVariableOp_5�Adam/AssignSubVariableOp_6�Adam/AssignSubVariableOp_7�Adam/AssignSubVariableOp_8�Adam/AssignSubVariableOp_9�Adam/ReadVariableOp�Adam/ReadVariableOp_1�Adam/ReadVariableOp_10�Adam/ReadVariableOp_11�Adam/ReadVariableOp_12�Adam/ReadVariableOp_13�Adam/ReadVariableOp_14�Adam/ReadVariableOp_15�Adam/ReadVariableOp_16�Adam/ReadVariableOp_17�Adam/ReadVariableOp_18�Adam/ReadVariableOp_19�Adam/ReadVariableOp_2�Adam/ReadVariableOp_20�Adam/ReadVariableOp_21�Adam/ReadVariableOp_22�Adam/ReadVariableOp_23�Adam/ReadVariableOp_24�Adam/ReadVariableOp_25�Adam/ReadVariableOp_26�Adam/ReadVariableOp_27�Adam/ReadVariableOp_28�Adam/ReadVariableOp_29�Adam/ReadVariableOp_3�Adam/ReadVariableOp_4�Adam/ReadVariableOp_5�Adam/ReadVariableOp_6�Adam/ReadVariableOp_7�Adam/ReadVariableOp_8�Adam/ReadVariableOp_9�Adam/Sqrt_1/ReadVariableOp�Adam/Sqrt_11/ReadVariableOp�Adam/Sqrt_13/ReadVariableOp�Adam/Sqrt_15/ReadVariableOp�Adam/Sqrt_17/ReadVariableOp�Adam/Sqrt_19/ReadVariableOp�Adam/Sqrt_3/ReadVariableOp�Adam/Sqrt_5/ReadVariableOp�Adam/Sqrt_7/ReadVariableOp�Adam/Sqrt_9/ReadVariableOp�Adam/sub_10/ReadVariableOp�Adam/sub_11/ReadVariableOp�Adam/sub_14/ReadVariableOp�Adam/sub_15/ReadVariableOp�Adam/sub_18/ReadVariableOp�Adam/sub_19/ReadVariableOp�Adam/sub_2/ReadVariableOp�Adam/sub_22/ReadVariableOp�Adam/sub_23/ReadVariableOp�Adam/sub_26/ReadVariableOp�Adam/sub_27/ReadVariableOp�Adam/sub_3/ReadVariableOp�Adam/sub_30/ReadVariableOp�Adam/sub_31/ReadVariableOp�Adam/sub_34/ReadVariableOp�Adam/sub_35/ReadVariableOp�Adam/sub_38/ReadVariableOp�Adam/sub_39/ReadVariableOp�Adam/sub_6/ReadVariableOp�Adam/sub_7/ReadVariableOp�AssignAddVariableOp�AssignAddVariableOp_1�AssignAddVariableOp_2�AssignAddVariableOp_3�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1�div_no_nan_1/ReadVariableOp�div_no_nan_1/ReadVariableOp_1�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0|
model/dense/MatMulMatMulx)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������`
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout/dropout/MulMul model/dense_1/Relu:activations:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������y
model/dropout/dropout/ShapeShape model/dense_1/Relu:activations:0*
T0*
_output_shapes
::���
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0i
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������b
model/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
model/dropout/dropout/SelectV2SelectV2&model/dropout/dropout/GreaterEqual:z:0model/dropout/dropout/Mul:z:0&model/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul'model/dropout/dropout/SelectV2:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������b
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_1/dropout/MulMul model/dense_2/Relu:activations:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������{
model/dropout_1/dropout/ShapeShape model/dense_2/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0k
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������d
model/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_1/dropout/SelectV2SelectV2(model/dropout_1/dropout/GreaterEqual:z:0model/dropout_1/dropout/Mul:z:0(model/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_3/MatMulMatMul)model/dropout_1/dropout/SelectV2:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@b
model/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/dropout_2/dropout/MulMul model/dense_3/Relu:activations:0&model/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@{
model/dropout_2/dropout/ShapeShape model/dense_3/Relu:activations:0*
T0*
_output_shapes
::���
4model/dropout_2/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0k
&model/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
$model/dropout_2/dropout/GreaterEqualGreaterEqual=model/dropout_2/dropout/random_uniform/RandomUniform:output:0/model/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@d
model/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/dropout_2/dropout/SelectV2SelectV2(model/dropout_2/dropout/GreaterEqual:z:0model/dropout_2/dropout/Mul:z:0(model/dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_4/MatMulMatMul)model/dropout_2/dropout/SelectV2:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_4/SigmoidSigmoidmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������^
binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,binary_crossentropy/logistic_loss/zeros_like	ZerosLikemodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqualmodel/dense_4/BiasAdd:output:00binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*'
_output_shapes
:����������
(binary_crossentropy/logistic_loss/SelectSelect2binary_crossentropy/logistic_loss/GreaterEqual:z:0model/dense_4/BiasAdd:output:00binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*'
_output_shapes
:���������~
%binary_crossentropy/logistic_loss/NegNegmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
*binary_crossentropy/logistic_loss/Select_1Select2binary_crossentropy/logistic_loss/GreaterEqual:z:0)binary_crossentropy/logistic_loss/Neg:y:0model/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
%binary_crossentropy/logistic_loss/mulMulmodel/dense_4/BiasAdd:output:0y*
T0*'
_output_shapes
:����������
%binary_crossentropy/logistic_loss/subSub1binary_crossentropy/logistic_loss/Select:output:0)binary_crossentropy/logistic_loss/mul:z:0*
T0*'
_output_shapes
:����������
%binary_crossentropy/logistic_loss/ExpExp3binary_crossentropy/logistic_loss/Select_1:output:0*
T0*'
_output_shapes
:����������
'binary_crossentropy/logistic_loss/Log1pLog1p)binary_crossentropy/logistic_loss/Exp:y:0*
T0*'
_output_shapes
:����������
!binary_crossentropy/logistic_lossAddV2)binary_crossentropy/logistic_loss/sub:z:0+binary_crossentropy/logistic_loss/Log1p:y:0*
T0*'
_output_shapes
:���������u
*binary_crossentropy/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
binary_crossentropy/MeanMean%binary_crossentropy/logistic_loss:z:03binary_crossentropy/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������l
'binary_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%binary_crossentropy/weighted_loss/MulMul!binary_crossentropy/Mean:output:00binary_crossentropy/weighted_loss/Const:output:0*
T0*#
_output_shapes
:���������s
)binary_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%binary_crossentropy/weighted_loss/SumSum)binary_crossentropy/weighted_loss/Mul:z:02binary_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: �
.binary_crossentropy/weighted_loss/num_elementsSize)binary_crossentropy/weighted_loss/Mul:z:0*
T0*
_output_shapes
: �
3binary_crossentropy/weighted_loss/num_elements/CastCast7binary_crossentropy/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: h
&binary_crossentropy/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : o
-binary_crossentropy/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : o
-binary_crossentropy/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
'binary_crossentropy/weighted_loss/rangeRange6binary_crossentropy/weighted_loss/range/start:output:0/binary_crossentropy/weighted_loss/Rank:output:06binary_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: �
'binary_crossentropy/weighted_loss/Sum_1Sum.binary_crossentropy/weighted_loss/Sum:output:00binary_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: �
'binary_crossentropy/weighted_loss/valueDivNoNan0binary_crossentropy/weighted_loss/Sum_1:output:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: D
ShapeShapey*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: b
MulMul+binary_crossentropy/weighted_loss/value:z:0Cast:y:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: l
SumSumMul:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: I
Sum_1SumCast:y:0range_1:output:0*
T0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum_1:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype0I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nanDivNoNanones:output:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
9gradient_tape/binary_crossentropy/weighted_loss/value/NegNeg0binary_crossentropy/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: �
Bgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_1DivNoNan=gradient_tape/binary_crossentropy/weighted_loss/value/Neg:y:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
Bgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_2DivNoNanFgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_1:z:07binary_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
9gradient_tape/binary_crossentropy/weighted_loss/value/mulMulones:output:0Fgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: ~
;gradient_tape/binary_crossentropy/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB �
=gradient_tape/binary_crossentropy/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
=gradient_tape/binary_crossentropy/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB �
?gradient_tape/binary_crossentropy/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB �
7gradient_tape/binary_crossentropy/weighted_loss/ReshapeReshapeDgradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan:z:0Hgradient_tape/binary_crossentropy/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: x
5gradient_tape/binary_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB �
4gradient_tape/binary_crossentropy/weighted_loss/TileTile@gradient_tape/binary_crossentropy/weighted_loss/Reshape:output:0>gradient_tape/binary_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: �
?gradient_tape/binary_crossentropy/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
9gradient_tape/binary_crossentropy/weighted_loss/Reshape_1Reshape=gradient_tape/binary_crossentropy/weighted_loss/Tile:output:0Hgradient_tape/binary_crossentropy/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:�
5gradient_tape/binary_crossentropy/weighted_loss/ShapeShape)binary_crossentropy/weighted_loss/Mul:z:0*
T0*
_output_shapes
::���
6gradient_tape/binary_crossentropy/weighted_loss/Tile_1TileBgradient_tape/binary_crossentropy/weighted_loss/Reshape_1:output:0>gradient_tape/binary_crossentropy/weighted_loss/Shape:output:0*
T0*#
_output_shapes
:����������
3gradient_tape/binary_crossentropy/weighted_loss/MulMul?gradient_tape/binary_crossentropy/weighted_loss/Tile_1:output:00binary_crossentropy/weighted_loss/Const:output:0*
T0*#
_output_shapes
:����������
'gradient_tape/binary_crossentropy/ShapeShape%binary_crossentropy/logistic_loss:z:0*
T0*
_output_shapes
::���
&gradient_tape/binary_crossentropy/SizeConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B :�
%gradient_tape/binary_crossentropy/addAddV23binary_crossentropy/Mean/reduction_indices:output:0/gradient_tape/binary_crossentropy/Size:output:0*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: �
%gradient_tape/binary_crossentropy/modFloorMod)gradient_tape/binary_crossentropy/add:z:0/gradient_tape/binary_crossentropy/Size:output:0*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: �
)gradient_tape/binary_crossentropy/Shape_1Const*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
valueB �
-gradient_tape/binary_crossentropy/range/startConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B : �
-gradient_tape/binary_crossentropy/range/deltaConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B :�
'gradient_tape/binary_crossentropy/rangeRange6gradient_tape/binary_crossentropy/range/start:output:0/gradient_tape/binary_crossentropy/Size:output:06gradient_tape/binary_crossentropy/range/delta:output:0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
:�
,gradient_tape/binary_crossentropy/ones/ConstConst*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: *
dtype0*
value	B :�
&gradient_tape/binary_crossentropy/onesFill2gradient_tape/binary_crossentropy/Shape_1:output:05gradient_tape/binary_crossentropy/ones/Const:output:0*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
: �
/gradient_tape/binary_crossentropy/DynamicStitchDynamicStitch0gradient_tape/binary_crossentropy/range:output:0)gradient_tape/binary_crossentropy/mod:z:00gradient_tape/binary_crossentropy/Shape:output:0/gradient_tape/binary_crossentropy/ones:output:0*
N*
T0*:
_class0
.,loc:@gradient_tape/binary_crossentropy/Shape*
_output_shapes
:�
)gradient_tape/binary_crossentropy/ReshapeReshape7gradient_tape/binary_crossentropy/weighted_loss/Mul:z:08gradient_tape/binary_crossentropy/DynamicStitch:merged:0*
T0*0
_output_shapes
:�������������������
-gradient_tape/binary_crossentropy/BroadcastToBroadcastTo2gradient_tape/binary_crossentropy/Reshape:output:00gradient_tape/binary_crossentropy/Shape:output:0*
T0*'
_output_shapes
:����������
)gradient_tape/binary_crossentropy/Shape_2Shape%binary_crossentropy/logistic_loss:z:0*
T0*
_output_shapes
::��j
(gradient_tape/binary_crossentropy/Size_1Const*
_output_shapes
: *
dtype0*
value	B :�
'gradient_tape/binary_crossentropy/add_1AddV23binary_crossentropy/Mean/reduction_indices:output:01gradient_tape/binary_crossentropy/Size_1:output:0*
T0*
_output_shapes
: �
'gradient_tape/binary_crossentropy/mod_1FloorMod+gradient_tape/binary_crossentropy/add_1:z:01gradient_tape/binary_crossentropy/Size_1:output:0*
T0*
_output_shapes
: q
/gradient_tape/binary_crossentropy/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*gradient_tape/binary_crossentropy/GatherV2GatherV22gradient_tape/binary_crossentropy/Shape_2:output:0+gradient_tape/binary_crossentropy/mod_1:z:08gradient_tape/binary_crossentropy/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: h
&gradient_tape/binary_crossentropy/RankConst*
_output_shapes
: *
dtype0*
value	B : q
/gradient_tape/binary_crossentropy/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : q
/gradient_tape/binary_crossentropy/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
)gradient_tape/binary_crossentropy/range_1Range8gradient_tape/binary_crossentropy/range_1/start:output:0/gradient_tape/binary_crossentropy/Rank:output:08gradient_tape/binary_crossentropy/range_1/delta:output:0*
_output_shapes
: �
&gradient_tape/binary_crossentropy/ProdProd3gradient_tape/binary_crossentropy/GatherV2:output:02gradient_tape/binary_crossentropy/range_1:output:0*
T0*
_output_shapes
: �
&gradient_tape/binary_crossentropy/CastCast/gradient_tape/binary_crossentropy/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
)gradient_tape/binary_crossentropy/truedivRealDiv6gradient_tape/binary_crossentropy/BroadcastTo:output:0*gradient_tape/binary_crossentropy/Cast:y:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/ShapeShape)binary_crossentropy/logistic_loss/sub:z:0*
T0*
_output_shapes
::���
7gradient_tape/binary_crossentropy/logistic_loss/Shape_1Shape+binary_crossentropy/logistic_loss/Log1p:y:0*
T0*
_output_shapes
::���
Egradient_tape/binary_crossentropy/logistic_loss/BroadcastGradientArgsBroadcastGradientArgs>gradient_tape/binary_crossentropy/logistic_loss/Shape:output:0@gradient_tape/binary_crossentropy/logistic_loss/Shape_1:output:0*2
_output_shapes 
:���������:����������
3gradient_tape/binary_crossentropy/logistic_loss/SumSum-gradient_tape/binary_crossentropy/truediv:z:0Jgradient_tape/binary_crossentropy/logistic_loss/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
7gradient_tape/binary_crossentropy/logistic_loss/ReshapeReshape<gradient_tape/binary_crossentropy/logistic_loss/Sum:output:0>gradient_tape/binary_crossentropy/logistic_loss/Shape:output:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/Sum_1Sum-gradient_tape/binary_crossentropy/truediv:z:0Jgradient_tape/binary_crossentropy/logistic_loss/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
9gradient_tape/binary_crossentropy/logistic_loss/Reshape_1Reshape>gradient_tape/binary_crossentropy/logistic_loss/Sum_1:output:0@gradient_tape/binary_crossentropy/logistic_loss/Shape_1:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:����������
7gradient_tape/binary_crossentropy/logistic_loss/sub/NegNeg@gradient_tape/binary_crossentropy/logistic_loss/Reshape:output:0*
T0*'
_output_shapes
:����������
9gradient_tape/binary_crossentropy/logistic_loss/sub/ShapeShape1binary_crossentropy/logistic_loss/Select:output:0*
T0*
_output_shapes
::���
;gradient_tape/binary_crossentropy/logistic_loss/sub/Shape_1Shape)binary_crossentropy/logistic_loss/mul:z:0*
T0*
_output_shapes
::���
Igradient_tape/binary_crossentropy/logistic_loss/sub/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/binary_crossentropy/logistic_loss/sub/Shape:output:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Shape_1:output:0*2
_output_shapes 
:���������:����������
7gradient_tape/binary_crossentropy/logistic_loss/sub/SumSum@gradient_tape/binary_crossentropy/logistic_loss/Reshape:output:0Ngradient_tape/binary_crossentropy/logistic_loss/sub/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
;gradient_tape/binary_crossentropy/logistic_loss/sub/ReshapeReshape@gradient_tape/binary_crossentropy/logistic_loss/sub/Sum:output:0Bgradient_tape/binary_crossentropy/logistic_loss/sub/Shape:output:0*
T0*'
_output_shapes
:����������
9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1Sum;gradient_tape/binary_crossentropy/logistic_loss/sub/Neg:y:0Ngradient_tape/binary_crossentropy/logistic_loss/sub/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
=gradient_tape/binary_crossentropy/logistic_loss/sub/Reshape_1ReshapeBgradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1:output:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Shape_1:output:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/add/xConst:^gradient_tape/binary_crossentropy/logistic_loss/Reshape_1*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3gradient_tape/binary_crossentropy/logistic_loss/addAddV2>gradient_tape/binary_crossentropy/logistic_loss/add/x:output:0)binary_crossentropy/logistic_loss/Exp:y:0*
T0*'
_output_shapes
:����������
:gradient_tape/binary_crossentropy/logistic_loss/Reciprocal
Reciprocal7gradient_tape/binary_crossentropy/logistic_loss/add:z:0*
T0*'
_output_shapes
:����������
3gradient_tape/binary_crossentropy/logistic_loss/mulMulBgradient_tape/binary_crossentropy/logistic_loss/Reshape_1:output:0>gradient_tape/binary_crossentropy/logistic_loss/Reciprocal:y:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:����������
:gradient_tape/binary_crossentropy/logistic_loss/zeros_like	ZerosLikemodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
6gradient_tape/binary_crossentropy/logistic_loss/SelectSelect2binary_crossentropy/logistic_loss/GreaterEqual:z:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Reshape:output:0>gradient_tape/binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*'
_output_shapes
:����������
8gradient_tape/binary_crossentropy/logistic_loss/Select_1Select2binary_crossentropy/logistic_loss/GreaterEqual:z:0>gradient_tape/binary_crossentropy/logistic_loss/zeros_like:y:0Dgradient_tape/binary_crossentropy/logistic_loss/sub/Reshape:output:0*
T0*'
_output_shapes
:����������
7gradient_tape/binary_crossentropy/logistic_loss/mul/MulMulFgradient_tape/binary_crossentropy/logistic_loss/sub/Reshape_1:output:0y*
T0*'
_output_shapes
:����������
9gradient_tape/binary_crossentropy/logistic_loss/mul/ShapeShapemodel/dense_4/BiasAdd:output:0*
T0*
_output_shapes
::��z
;gradient_tape/binary_crossentropy/logistic_loss/mul/Shape_1Shapey*
T0*
_output_shapes
::���
Igradient_tape/binary_crossentropy/logistic_loss/mul/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/binary_crossentropy/logistic_loss/mul/Shape:output:0Dgradient_tape/binary_crossentropy/logistic_loss/mul/Shape_1:output:0*2
_output_shapes 
:���������:����������
7gradient_tape/binary_crossentropy/logistic_loss/mul/SumSum;gradient_tape/binary_crossentropy/logistic_loss/mul/Mul:z:0Ngradient_tape/binary_crossentropy/logistic_loss/mul/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
;gradient_tape/binary_crossentropy/logistic_loss/mul/ReshapeReshape@gradient_tape/binary_crossentropy/logistic_loss/mul/Sum:output:0Bgradient_tape/binary_crossentropy/logistic_loss/mul/Shape:output:0*
T0*'
_output_shapes
:����������
5gradient_tape/binary_crossentropy/logistic_loss/mul_1Mul7gradient_tape/binary_crossentropy/logistic_loss/mul:z:0)binary_crossentropy/logistic_loss/Exp:y:0*
T0*'
_output_shapes
:����������
<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1	ZerosLike)binary_crossentropy/logistic_loss/Neg:y:0*
T0*'
_output_shapes
:����������
8gradient_tape/binary_crossentropy/logistic_loss/Select_2Select2binary_crossentropy/logistic_loss/GreaterEqual:z:09gradient_tape/binary_crossentropy/logistic_loss/mul_1:z:0@gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1:y:0*
T0*'
_output_shapes
:����������
8gradient_tape/binary_crossentropy/logistic_loss/Select_3Select2binary_crossentropy/logistic_loss/GreaterEqual:z:0@gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1:y:09gradient_tape/binary_crossentropy/logistic_loss/mul_1:z:0*
T0*'
_output_shapes
:����������
3gradient_tape/binary_crossentropy/logistic_loss/NegNegAgradient_tape/binary_crossentropy/logistic_loss/Select_2:output:0*
T0*'
_output_shapes
:����������
AddNAddN?gradient_tape/binary_crossentropy/logistic_loss/Select:output:0Dgradient_tape/binary_crossentropy/logistic_loss/mul/Reshape:output:0Agradient_tape/binary_crossentropy/logistic_loss/Select_3:output:07gradient_tape/binary_crossentropy/logistic_loss/Neg:y:0*
N*
T0*'
_output_shapes
:���������o
/gradient_tape/model/dense_4/BiasAdd/BiasAddGradBiasAddGrad
AddN:sum:0*
T0*
_output_shapes
:�
)gradient_tape/model/dense_4/MatMul/MatMulMatMul
AddN:sum:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_4/MatMul/MatMul_1MatMul)model/dropout_2/dropout/SelectV2:output:0
AddN:sum:0*
T0*
_output_shapes

:@*
grad_b(*
transpose_a(p
+gradient_tape/model/dropout_2/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.gradient_tape/model/dropout_2/dropout/SelectV2SelectV2(model/dropout_2/dropout/GreaterEqual:z:03gradient_tape/model/dense_4/MatMul/MatMul:product:04gradient_tape/model/dropout_2/dropout/zeros:output:0*
T0*'
_output_shapes
:���������@�
0gradient_tape/model/dropout_2/dropout/SelectV2_1SelectV2(model/dropout_2/dropout/GreaterEqual:z:04gradient_tape/model/dropout_2/dropout/zeros:output:03gradient_tape/model/dense_4/MatMul/MatMul:product:0*
T0*'
_output_shapes
:���������@�
+gradient_tape/model/dropout_2/dropout/ShapeShapemodel/dropout_2/dropout/Mul:z:0*
T0*
_output_shapes
::���
-gradient_tape/model/dropout_2/dropout/Shape_1Shape)model/dropout_2/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
;gradient_tape/model/dropout_2/dropout/BroadcastGradientArgsBroadcastGradientArgs4gradient_tape/model/dropout_2/dropout/Shape:output:06gradient_tape/model/dropout_2/dropout/Shape_1:output:0*2
_output_shapes 
:���������:����������
)gradient_tape/model/dropout_2/dropout/SumSum7gradient_tape/model/dropout_2/dropout/SelectV2:output:0@gradient_tape/model/dropout_2/dropout/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
-gradient_tape/model/dropout_2/dropout/ReshapeReshape2gradient_tape/model/dropout_2/dropout/Sum:output:04gradient_tape/model/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@p
-gradient_tape/model/dropout_2/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB �
-gradient_tape/model/dropout_2/dropout/Shape_3Shape)model/dropout_2/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
=gradient_tape/model/dropout_2/dropout/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
+gradient_tape/model/dropout_2/dropout/Sum_1Sum9gradient_tape/model/dropout_2/dropout/SelectV2_1:output:0Fgradient_tape/model/dropout_2/dropout/Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
/gradient_tape/model/dropout_2/dropout/Reshape_1Reshape4gradient_tape/model/dropout_2/dropout/Sum_1:output:06gradient_tape/model/dropout_2/dropout/Shape_2:output:0*
T0*
_output_shapes
: �
)gradient_tape/model/dropout_2/dropout/MulMul6gradient_tape/model/dropout_2/dropout/Reshape:output:0&model/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:���������@�
$gradient_tape/model/dense_3/ReluGradReluGrad-gradient_tape/model/dropout_2/dropout/Mul:z:0 model/dense_3/Relu:activations:0*
T0*'
_output_shapes
:���������@�
/gradient_tape/model/dense_3/BiasAdd/BiasAddGradBiasAddGrad0gradient_tape/model/dense_3/ReluGrad:backprops:0*
T0*
_output_shapes
:@�
)gradient_tape/model/dense_3/MatMul/MatMulMatMul0gradient_tape/model/dense_3/ReluGrad:backprops:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_3/MatMul/MatMul_1MatMul)model/dropout_1/dropout/SelectV2:output:00gradient_tape/model/dense_3/ReluGrad:backprops:0*
T0*
_output_shapes
:	�@*
grad_b(*
transpose_a(p
+gradient_tape/model/dropout_1/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.gradient_tape/model/dropout_1/dropout/SelectV2SelectV2(model/dropout_1/dropout/GreaterEqual:z:03gradient_tape/model/dense_3/MatMul/MatMul:product:04gradient_tape/model/dropout_1/dropout/zeros:output:0*
T0*(
_output_shapes
:�����������
0gradient_tape/model/dropout_1/dropout/SelectV2_1SelectV2(model/dropout_1/dropout/GreaterEqual:z:04gradient_tape/model/dropout_1/dropout/zeros:output:03gradient_tape/model/dense_3/MatMul/MatMul:product:0*
T0*(
_output_shapes
:�����������
+gradient_tape/model/dropout_1/dropout/ShapeShapemodel/dropout_1/dropout/Mul:z:0*
T0*
_output_shapes
::���
-gradient_tape/model/dropout_1/dropout/Shape_1Shape)model/dropout_1/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
;gradient_tape/model/dropout_1/dropout/BroadcastGradientArgsBroadcastGradientArgs4gradient_tape/model/dropout_1/dropout/Shape:output:06gradient_tape/model/dropout_1/dropout/Shape_1:output:0*2
_output_shapes 
:���������:����������
)gradient_tape/model/dropout_1/dropout/SumSum7gradient_tape/model/dropout_1/dropout/SelectV2:output:0@gradient_tape/model/dropout_1/dropout/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
-gradient_tape/model/dropout_1/dropout/ReshapeReshape2gradient_tape/model/dropout_1/dropout/Sum:output:04gradient_tape/model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������p
-gradient_tape/model/dropout_1/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB �
-gradient_tape/model/dropout_1/dropout/Shape_3Shape)model/dropout_1/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
=gradient_tape/model/dropout_1/dropout/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
+gradient_tape/model/dropout_1/dropout/Sum_1Sum9gradient_tape/model/dropout_1/dropout/SelectV2_1:output:0Fgradient_tape/model/dropout_1/dropout/Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
/gradient_tape/model/dropout_1/dropout/Reshape_1Reshape4gradient_tape/model/dropout_1/dropout/Sum_1:output:06gradient_tape/model/dropout_1/dropout/Shape_2:output:0*
T0*
_output_shapes
: �
)gradient_tape/model/dropout_1/dropout/MulMul6gradient_tape/model/dropout_1/dropout/Reshape:output:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$gradient_tape/model/dense_2/ReluGradReluGrad-gradient_tape/model/dropout_1/dropout/Mul:z:0 model/dense_2/Relu:activations:0*
T0*(
_output_shapes
:�����������
/gradient_tape/model/dense_2/BiasAdd/BiasAddGradBiasAddGrad0gradient_tape/model/dense_2/ReluGrad:backprops:0*
T0*
_output_shapes	
:��
)gradient_tape/model/dense_2/MatMul/MatMulMatMul0gradient_tape/model/dense_2/ReluGrad:backprops:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_2/MatMul/MatMul_1MatMul'model/dropout/dropout/SelectV2:output:00gradient_tape/model/dense_2/ReluGrad:backprops:0*
T0* 
_output_shapes
:
��*
grad_b(*
transpose_a(n
)gradient_tape/model/dropout/dropout/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,gradient_tape/model/dropout/dropout/SelectV2SelectV2&model/dropout/dropout/GreaterEqual:z:03gradient_tape/model/dense_2/MatMul/MatMul:product:02gradient_tape/model/dropout/dropout/zeros:output:0*
T0*(
_output_shapes
:�����������
.gradient_tape/model/dropout/dropout/SelectV2_1SelectV2&model/dropout/dropout/GreaterEqual:z:02gradient_tape/model/dropout/dropout/zeros:output:03gradient_tape/model/dense_2/MatMul/MatMul:product:0*
T0*(
_output_shapes
:�����������
)gradient_tape/model/dropout/dropout/ShapeShapemodel/dropout/dropout/Mul:z:0*
T0*
_output_shapes
::���
+gradient_tape/model/dropout/dropout/Shape_1Shape'model/dropout/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
9gradient_tape/model/dropout/dropout/BroadcastGradientArgsBroadcastGradientArgs2gradient_tape/model/dropout/dropout/Shape:output:04gradient_tape/model/dropout/dropout/Shape_1:output:0*2
_output_shapes 
:���������:����������
'gradient_tape/model/dropout/dropout/SumSum5gradient_tape/model/dropout/dropout/SelectV2:output:0>gradient_tape/model/dropout/dropout/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:������������������*
	keep_dims(�
+gradient_tape/model/dropout/dropout/ReshapeReshape0gradient_tape/model/dropout/dropout/Sum:output:02gradient_tape/model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������n
+gradient_tape/model/dropout/dropout/Shape_2Const*
_output_shapes
: *
dtype0*
valueB �
+gradient_tape/model/dropout/dropout/Shape_3Shape'model/dropout/dropout/SelectV2:output:0*
T0*
_output_shapes
::���
;gradient_tape/model/dropout/dropout/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
)gradient_tape/model/dropout/dropout/Sum_1Sum7gradient_tape/model/dropout/dropout/SelectV2_1:output:0Dgradient_tape/model/dropout/dropout/Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-gradient_tape/model/dropout/dropout/Reshape_1Reshape2gradient_tape/model/dropout/dropout/Sum_1:output:04gradient_tape/model/dropout/dropout/Shape_2:output:0*
T0*
_output_shapes
: �
'gradient_tape/model/dropout/dropout/MulMul4gradient_tape/model/dropout/dropout/Reshape:output:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$gradient_tape/model/dense_1/ReluGradReluGrad+gradient_tape/model/dropout/dropout/Mul:z:0 model/dense_1/Relu:activations:0*
T0*(
_output_shapes
:�����������
/gradient_tape/model/dense_1/BiasAdd/BiasAddGradBiasAddGrad0gradient_tape/model/dense_1/ReluGrad:backprops:0*
T0*
_output_shapes	
:��
)gradient_tape/model/dense_1/MatMul/MatMulMatMul0gradient_tape/model/dense_1/ReluGrad:backprops:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
grad_a(*
transpose_b(�
+gradient_tape/model/dense_1/MatMul/MatMul_1MatMulmodel/dense/Relu:activations:00gradient_tape/model/dense_1/ReluGrad:backprops:0*
T0*
_output_shapes
:	�*
grad_b(*
transpose_a(�
"gradient_tape/model/dense/ReluGradReluGrad3gradient_tape/model/dense_1/MatMul/MatMul:product:0model/dense/Relu:activations:0*
T0*'
_output_shapes
:����������
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGrad.gradient_tape/model/dense/ReluGrad:backprops:0*
T0*
_output_shapes
:�
'gradient_tape/model/dense/MatMul/MatMulMatMulx.gradient_tape/model/dense/ReluGrad:backprops:0*
T0*
_output_shapes

:*
grad_b(*
transpose_a(p
IdentityIdentity1gradient_tape/model/dense/MatMul/MatMul:product:0*
T0*
_output_shapes

:s

Identity_1Identity6gradient_tape/model/dense/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:w

Identity_2Identity5gradient_tape/model/dense_1/MatMul/MatMul_1:product:0*
T0*
_output_shapes
:	�v

Identity_3Identity8gradient_tape/model/dense_1/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:�x

Identity_4Identity5gradient_tape/model/dense_2/MatMul/MatMul_1:product:0*
T0* 
_output_shapes
:
��v

Identity_5Identity8gradient_tape/model/dense_2/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes	
:�w

Identity_6Identity5gradient_tape/model/dense_3/MatMul/MatMul_1:product:0*
T0*
_output_shapes
:	�@u

Identity_7Identity8gradient_tape/model/dense_3/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:@v

Identity_8Identity5gradient_tape/model/dense_4/MatMul/MatMul_1:product:0*
T0*
_output_shapes

:@u

Identity_9Identity8gradient_tape/model/dense_4/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:�
	IdentityN	IdentityN1gradient_tape/model/dense/MatMul/MatMul:product:06gradient_tape/model/dense/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_1/MatMul/MatMul_1:product:08gradient_tape/model/dense_1/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_2/MatMul/MatMul_1:product:08gradient_tape/model/dense_2/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_3/MatMul/MatMul_1:product:08gradient_tape/model/dense_3/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_4/MatMul/MatMul_1:product:08gradient_tape/model/dense_4/BiasAdd/BiasAddGrad:output:01gradient_tape/model/dense/MatMul/MatMul:product:06gradient_tape/model/dense/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_1/MatMul/MatMul_1:product:08gradient_tape/model/dense_1/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_2/MatMul/MatMul_1:product:08gradient_tape/model/dense_2/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_3/MatMul/MatMul_1:product:08gradient_tape/model/dense_3/BiasAdd/BiasAddGrad:output:05gradient_tape/model/dense_4/MatMul/MatMul_1:product:08gradient_tape/model/dense_4/BiasAdd/BiasAddGrad:output:0*
T
2**
_gradient_op_typeCustomGradient-1317*�
_output_shapes�
�:::	�:�:
��:�:	�@:@:@::::	�:�:
��:�:	�@:@:@:h
Adam/ReadVariableOpReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	L

Adam/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd
Adam/addAddV2Adam/ReadVariableOp:value:0Adam/add/y:output:0*
T0	*
_output_shapes
: O
	Adam/CastCastAdam/add:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
Adam/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
Adam/PowPowAdam/Cast_1/x:output:0Adam/Cast:y:0*
T0*
_output_shapes
: R
Adam/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?Y

Adam/Pow_1PowAdam/Cast_2/x:output:0Adam/Cast:y:0*
T0*
_output_shapes
: O

Adam/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
Adam/subSubAdam/sub/x:output:0Adam/Pow_1:z:0*
T0*
_output_shapes
: @
	Adam/SqrtSqrtAdam/sub:z:0*
T0*
_output_shapes
: l
Adam/ReadVariableOp_1ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0^
Adam/mulMulAdam/ReadVariableOp_1:value:0Adam/Sqrt:y:0*
T0*
_output_shapes
: Q
Adam/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?W

Adam/sub_1SubAdam/sub_1/x:output:0Adam/Pow:z:0*
T0*
_output_shapes
: V
Adam/truedivRealDivAdam/mul:z:0Adam/sub_1:z:0*
T0*
_output_shapes
: |
Adam/sub_2/ReadVariableOpReadVariableOp"adam_sub_2_readvariableop_resource*
_output_shapes

:*
dtype0q

Adam/sub_2SubIdentityN:output:0!Adam/sub_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Q
Adam/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=a

Adam/mul_1MulAdam/sub_2:z:0Adam/mul_1/y:output:0*
T0*
_output_shapes

:�
Adam/AssignAddVariableOpAssignAddVariableOp"adam_sub_2_readvariableop_resourceAdam/mul_1:z:0^Adam/sub_2/ReadVariableOp*
_output_shapes
 *
dtype0R
Adam/SquareSquareIdentityN:output:0*
T0*
_output_shapes

:|
Adam/sub_3/ReadVariableOpReadVariableOp"adam_sub_3_readvariableop_resource*
_output_shapes

:*
dtype0n

Adam/sub_3SubAdam/Square:y:0!Adam/sub_3/ReadVariableOp:value:0*
T0*
_output_shapes

:Q
Adam/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a

Adam/mul_2MulAdam/sub_3:z:0Adam/mul_2/y:output:0*
T0*
_output_shapes

:�
Adam/AssignAddVariableOp_1AssignAddVariableOp"adam_sub_3_readvariableop_resourceAdam/mul_2:z:0^Adam/sub_3/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_2ReadVariableOp"adam_sub_2_readvariableop_resource^Adam/AssignAddVariableOp*
_output_shapes

:*
dtype0k

Adam/mul_3MulAdam/ReadVariableOp_2:value:0Adam/truediv:z:0*
T0*
_output_shapes

:�
Adam/Sqrt_1/ReadVariableOpReadVariableOp"adam_sub_3_readvariableop_resource^Adam/AssignAddVariableOp_1*
_output_shapes

:*
dtype0`
Adam/Sqrt_1Sqrt"Adam/Sqrt_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Q
Adam/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3d

Adam/add_1AddV2Adam/Sqrt_1:y:0Adam/add_1/y:output:0*
T0*
_output_shapes

:b
Adam/truediv_1RealDivAdam/mul_3:z:0Adam/add_1:z:0*
T0*
_output_shapes

:�
Adam/AssignSubVariableOpAssignSubVariableOp*model_dense_matmul_readvariableop_resourceAdam/truediv_1:z:0"^model/dense/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0j
Adam/ReadVariableOp_3ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rj

Adam/add_2AddV2Adam/ReadVariableOp_3:value:0Adam/add_2/y:output:0*
T0	*
_output_shapes
: S
Adam/Cast_3CastAdam/add_2:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
Adam/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?[

Adam/Pow_2PowAdam/Cast_4/x:output:0Adam/Cast_3:y:0*
T0*
_output_shapes
: R
Adam/Cast_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?[

Adam/Pow_3PowAdam/Cast_5/x:output:0Adam/Cast_3:y:0*
T0*
_output_shapes
: Q
Adam/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_4SubAdam/sub_4/x:output:0Adam/Pow_3:z:0*
T0*
_output_shapes
: D
Adam/Sqrt_2SqrtAdam/sub_4:z:0*
T0*
_output_shapes
: l
Adam/ReadVariableOp_4ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0b

Adam/mul_4MulAdam/ReadVariableOp_4:value:0Adam/Sqrt_2:y:0*
T0*
_output_shapes
: Q
Adam/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_5SubAdam/sub_5/x:output:0Adam/Pow_2:z:0*
T0*
_output_shapes
: Z
Adam/truediv_2RealDivAdam/mul_4:z:0Adam/sub_5:z:0*
T0*
_output_shapes
: x
Adam/sub_6/ReadVariableOpReadVariableOp"adam_sub_6_readvariableop_resource*
_output_shapes
:*
dtype0m

Adam/sub_6SubIdentityN:output:1!Adam/sub_6/ReadVariableOp:value:0*
T0*
_output_shapes
:Q
Adam/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=]

Adam/mul_5MulAdam/sub_6:z:0Adam/mul_5/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_2AssignAddVariableOp"adam_sub_6_readvariableop_resourceAdam/mul_5:z:0^Adam/sub_6/ReadVariableOp*
_output_shapes
 *
dtype0P
Adam/Square_1SquareIdentityN:output:1*
T0*
_output_shapes
:x
Adam/sub_7/ReadVariableOpReadVariableOp"adam_sub_7_readvariableop_resource*
_output_shapes
:*
dtype0l

Adam/sub_7SubAdam/Square_1:y:0!Adam/sub_7/ReadVariableOp:value:0*
T0*
_output_shapes
:Q
Adam/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:]

Adam/mul_6MulAdam/sub_7:z:0Adam/mul_6/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_3AssignAddVariableOp"adam_sub_7_readvariableop_resourceAdam/mul_6:z:0^Adam/sub_7/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_5ReadVariableOp"adam_sub_6_readvariableop_resource^Adam/AssignAddVariableOp_2*
_output_shapes
:*
dtype0i

Adam/mul_7MulAdam/ReadVariableOp_5:value:0Adam/truediv_2:z:0*
T0*
_output_shapes
:�
Adam/Sqrt_3/ReadVariableOpReadVariableOp"adam_sub_7_readvariableop_resource^Adam/AssignAddVariableOp_3*
_output_shapes
:*
dtype0\
Adam/Sqrt_3Sqrt"Adam/Sqrt_3/ReadVariableOp:value:0*
T0*
_output_shapes
:Q
Adam/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3`

Adam/add_3AddV2Adam/Sqrt_3:y:0Adam/add_3/y:output:0*
T0*
_output_shapes
:^
Adam/truediv_3RealDivAdam/mul_7:z:0Adam/add_3:z:0*
T0*
_output_shapes
:�
Adam/AssignSubVariableOp_1AssignSubVariableOp+model_dense_biasadd_readvariableop_resourceAdam/truediv_3:z:0#^model/dense/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0j
Adam/ReadVariableOp_6ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_4/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rj

Adam/add_4AddV2Adam/ReadVariableOp_6:value:0Adam/add_4/y:output:0*
T0	*
_output_shapes
: S
Adam/Cast_6CastAdam/add_4:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
Adam/Cast_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?[

Adam/Pow_4PowAdam/Cast_7/x:output:0Adam/Cast_6:y:0*
T0*
_output_shapes
: R
Adam/Cast_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?[

Adam/Pow_5PowAdam/Cast_8/x:output:0Adam/Cast_6:y:0*
T0*
_output_shapes
: Q
Adam/sub_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_8SubAdam/sub_8/x:output:0Adam/Pow_5:z:0*
T0*
_output_shapes
: D
Adam/Sqrt_4SqrtAdam/sub_8:z:0*
T0*
_output_shapes
: l
Adam/ReadVariableOp_7ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0b

Adam/mul_8MulAdam/ReadVariableOp_7:value:0Adam/Sqrt_4:y:0*
T0*
_output_shapes
: Q
Adam/sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

Adam/sub_9SubAdam/sub_9/x:output:0Adam/Pow_4:z:0*
T0*
_output_shapes
: Z
Adam/truediv_4RealDivAdam/mul_8:z:0Adam/sub_9:z:0*
T0*
_output_shapes
: 
Adam/sub_10/ReadVariableOpReadVariableOp#adam_sub_10_readvariableop_resource*
_output_shapes
:	�*
dtype0t
Adam/sub_10SubIdentityN:output:2"Adam/sub_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Q
Adam/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=c

Adam/mul_9MulAdam/sub_10:z:0Adam/mul_9/y:output:0*
T0*
_output_shapes
:	��
Adam/AssignAddVariableOp_4AssignAddVariableOp#adam_sub_10_readvariableop_resourceAdam/mul_9:z:0^Adam/sub_10/ReadVariableOp*
_output_shapes
 *
dtype0U
Adam/Square_2SquareIdentityN:output:2*
T0*
_output_shapes
:	�
Adam/sub_11/ReadVariableOpReadVariableOp#adam_sub_11_readvariableop_resource*
_output_shapes
:	�*
dtype0s
Adam/sub_11SubAdam/Square_2:y:0"Adam/sub_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�R
Adam/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:e
Adam/mul_10MulAdam/sub_11:z:0Adam/mul_10/y:output:0*
T0*
_output_shapes
:	��
Adam/AssignAddVariableOp_5AssignAddVariableOp#adam_sub_11_readvariableop_resourceAdam/mul_10:z:0^Adam/sub_11/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_8ReadVariableOp#adam_sub_10_readvariableop_resource^Adam/AssignAddVariableOp_4*
_output_shapes
:	�*
dtype0o
Adam/mul_11MulAdam/ReadVariableOp_8:value:0Adam/truediv_4:z:0*
T0*
_output_shapes
:	��
Adam/Sqrt_5/ReadVariableOpReadVariableOp#adam_sub_11_readvariableop_resource^Adam/AssignAddVariableOp_5*
_output_shapes
:	�*
dtype0a
Adam/Sqrt_5Sqrt"Adam/Sqrt_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Q
Adam/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e

Adam/add_5AddV2Adam/Sqrt_5:y:0Adam/add_5/y:output:0*
T0*
_output_shapes
:	�d
Adam/truediv_5RealDivAdam/mul_11:z:0Adam/add_5:z:0*
T0*
_output_shapes
:	��
Adam/AssignSubVariableOp_2AssignSubVariableOp,model_dense_1_matmul_readvariableop_resourceAdam/truediv_5:z:0$^model/dense_1/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0j
Adam/ReadVariableOp_9ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rj

Adam/add_6AddV2Adam/ReadVariableOp_9:value:0Adam/add_6/y:output:0*
T0	*
_output_shapes
: S
Adam/Cast_9CastAdam/add_6:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?\

Adam/Pow_6PowAdam/Cast_10/x:output:0Adam/Cast_9:y:0*
T0*
_output_shapes
: S
Adam/Cast_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?\

Adam/Pow_7PowAdam/Cast_11/x:output:0Adam/Cast_9:y:0*
T0*
_output_shapes
: R
Adam/sub_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_12SubAdam/sub_12/x:output:0Adam/Pow_7:z:0*
T0*
_output_shapes
: E
Adam/Sqrt_6SqrtAdam/sub_12:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_10ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0d
Adam/mul_12MulAdam/ReadVariableOp_10:value:0Adam/Sqrt_6:y:0*
T0*
_output_shapes
: R
Adam/sub_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_13SubAdam/sub_13/x:output:0Adam/Pow_6:z:0*
T0*
_output_shapes
: \
Adam/truediv_6RealDivAdam/mul_12:z:0Adam/sub_13:z:0*
T0*
_output_shapes
: {
Adam/sub_14/ReadVariableOpReadVariableOp#adam_sub_14_readvariableop_resource*
_output_shapes	
:�*
dtype0p
Adam/sub_14SubIdentityN:output:3"Adam/sub_14/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=a
Adam/mul_13MulAdam/sub_14:z:0Adam/mul_13/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_6AssignAddVariableOp#adam_sub_14_readvariableop_resourceAdam/mul_13:z:0^Adam/sub_14/ReadVariableOp*
_output_shapes
 *
dtype0Q
Adam/Square_3SquareIdentityN:output:3*
T0*
_output_shapes	
:�{
Adam/sub_15/ReadVariableOpReadVariableOp#adam_sub_15_readvariableop_resource*
_output_shapes	
:�*
dtype0o
Adam/sub_15SubAdam/Square_3:y:0"Adam/sub_15/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
Adam/mul_14MulAdam/sub_15:z:0Adam/mul_14/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_7AssignAddVariableOp#adam_sub_15_readvariableop_resourceAdam/mul_14:z:0^Adam/sub_15/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_11ReadVariableOp#adam_sub_14_readvariableop_resource^Adam/AssignAddVariableOp_6*
_output_shapes	
:�*
dtype0l
Adam/mul_15MulAdam/ReadVariableOp_11:value:0Adam/truediv_6:z:0*
T0*
_output_shapes	
:��
Adam/Sqrt_7/ReadVariableOpReadVariableOp#adam_sub_15_readvariableop_resource^Adam/AssignAddVariableOp_7*
_output_shapes	
:�*
dtype0]
Adam/Sqrt_7Sqrt"Adam/Sqrt_7/ReadVariableOp:value:0*
T0*
_output_shapes	
:�Q
Adam/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a

Adam/add_7AddV2Adam/Sqrt_7:y:0Adam/add_7/y:output:0*
T0*
_output_shapes	
:�`
Adam/truediv_7RealDivAdam/mul_15:z:0Adam/add_7:z:0*
T0*
_output_shapes	
:��
Adam/AssignSubVariableOp_3AssignSubVariableOp-model_dense_1_biasadd_readvariableop_resourceAdam/truediv_7:z:0%^model/dense_1/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_12ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	N
Adam/add_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rk

Adam/add_8AddV2Adam/ReadVariableOp_12:value:0Adam/add_8/y:output:0*
T0	*
_output_shapes
: T
Adam/Cast_12CastAdam/add_8:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]

Adam/Pow_8PowAdam/Cast_13/x:output:0Adam/Cast_12:y:0*
T0*
_output_shapes
: S
Adam/Cast_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?]

Adam/Pow_9PowAdam/Cast_14/x:output:0Adam/Cast_12:y:0*
T0*
_output_shapes
: R
Adam/sub_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_16SubAdam/sub_16/x:output:0Adam/Pow_9:z:0*
T0*
_output_shapes
: E
Adam/Sqrt_8SqrtAdam/sub_16:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_13ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0d
Adam/mul_16MulAdam/ReadVariableOp_13:value:0Adam/Sqrt_8:y:0*
T0*
_output_shapes
: R
Adam/sub_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
Adam/sub_17SubAdam/sub_17/x:output:0Adam/Pow_8:z:0*
T0*
_output_shapes
: \
Adam/truediv_8RealDivAdam/mul_16:z:0Adam/sub_17:z:0*
T0*
_output_shapes
: �
Adam/sub_18/ReadVariableOpReadVariableOp#adam_sub_18_readvariableop_resource* 
_output_shapes
:
��*
dtype0u
Adam/sub_18SubIdentityN:output:4"Adam/sub_18/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��R
Adam/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=f
Adam/mul_17MulAdam/sub_18:z:0Adam/mul_17/y:output:0*
T0* 
_output_shapes
:
���
Adam/AssignAddVariableOp_8AssignAddVariableOp#adam_sub_18_readvariableop_resourceAdam/mul_17:z:0^Adam/sub_18/ReadVariableOp*
_output_shapes
 *
dtype0V
Adam/Square_4SquareIdentityN:output:4*
T0* 
_output_shapes
:
���
Adam/sub_19/ReadVariableOpReadVariableOp#adam_sub_19_readvariableop_resource* 
_output_shapes
:
��*
dtype0t
Adam/sub_19SubAdam/Square_4:y:0"Adam/sub_19/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��R
Adam/mul_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:f
Adam/mul_18MulAdam/sub_19:z:0Adam/mul_18/y:output:0*
T0* 
_output_shapes
:
���
Adam/AssignAddVariableOp_9AssignAddVariableOp#adam_sub_19_readvariableop_resourceAdam/mul_18:z:0^Adam/sub_19/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_14ReadVariableOp#adam_sub_18_readvariableop_resource^Adam/AssignAddVariableOp_8* 
_output_shapes
:
��*
dtype0q
Adam/mul_19MulAdam/ReadVariableOp_14:value:0Adam/truediv_8:z:0*
T0* 
_output_shapes
:
���
Adam/Sqrt_9/ReadVariableOpReadVariableOp#adam_sub_19_readvariableop_resource^Adam/AssignAddVariableOp_9* 
_output_shapes
:
��*
dtype0b
Adam/Sqrt_9Sqrt"Adam/Sqrt_9/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��Q
Adam/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3f

Adam/add_9AddV2Adam/Sqrt_9:y:0Adam/add_9/y:output:0*
T0* 
_output_shapes
:
��e
Adam/truediv_9RealDivAdam/mul_19:z:0Adam/add_9:z:0*
T0* 
_output_shapes
:
���
Adam/AssignSubVariableOp_4AssignSubVariableOp,model_dense_2_matmul_readvariableop_resourceAdam/truediv_9:z:0$^model/dense_2/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_15ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_10/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_10AddV2Adam/ReadVariableOp_15:value:0Adam/add_10/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_15CastAdam/add_10:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_10PowAdam/Cast_16/x:output:0Adam/Cast_15:y:0*
T0*
_output_shapes
: S
Adam/Cast_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_11PowAdam/Cast_17/x:output:0Adam/Cast_15:y:0*
T0*
_output_shapes
: R
Adam/sub_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_20SubAdam/sub_20/x:output:0Adam/Pow_11:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_10SqrtAdam/sub_20:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_16ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_20MulAdam/ReadVariableOp_16:value:0Adam/Sqrt_10:y:0*
T0*
_output_shapes
: R
Adam/sub_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_21SubAdam/sub_21/x:output:0Adam/Pow_10:z:0*
T0*
_output_shapes
: ]
Adam/truediv_10RealDivAdam/mul_20:z:0Adam/sub_21:z:0*
T0*
_output_shapes
: {
Adam/sub_22/ReadVariableOpReadVariableOp#adam_sub_22_readvariableop_resource*
_output_shapes	
:�*
dtype0p
Adam/sub_22SubIdentityN:output:5"Adam/sub_22/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=a
Adam/mul_21MulAdam/sub_22:z:0Adam/mul_21/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_10AssignAddVariableOp#adam_sub_22_readvariableop_resourceAdam/mul_21:z:0^Adam/sub_22/ReadVariableOp*
_output_shapes
 *
dtype0Q
Adam/Square_5SquareIdentityN:output:5*
T0*
_output_shapes	
:�{
Adam/sub_23/ReadVariableOpReadVariableOp#adam_sub_23_readvariableop_resource*
_output_shapes	
:�*
dtype0o
Adam/sub_23SubAdam/Square_5:y:0"Adam/sub_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/mul_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
Adam/mul_22MulAdam/sub_23:z:0Adam/mul_22/y:output:0*
T0*
_output_shapes	
:��
Adam/AssignAddVariableOp_11AssignAddVariableOp#adam_sub_23_readvariableop_resourceAdam/mul_22:z:0^Adam/sub_23/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_17ReadVariableOp#adam_sub_22_readvariableop_resource^Adam/AssignAddVariableOp_10*
_output_shapes	
:�*
dtype0m
Adam/mul_23MulAdam/ReadVariableOp_17:value:0Adam/truediv_10:z:0*
T0*
_output_shapes	
:��
Adam/Sqrt_11/ReadVariableOpReadVariableOp#adam_sub_23_readvariableop_resource^Adam/AssignAddVariableOp_11*
_output_shapes	
:�*
dtype0_
Adam/Sqrt_11Sqrt#Adam/Sqrt_11/ReadVariableOp:value:0*
T0*
_output_shapes	
:�R
Adam/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3d
Adam/add_11AddV2Adam/Sqrt_11:y:0Adam/add_11/y:output:0*
T0*
_output_shapes	
:�b
Adam/truediv_11RealDivAdam/mul_23:z:0Adam/add_11:z:0*
T0*
_output_shapes	
:��
Adam/AssignSubVariableOp_5AssignSubVariableOp-model_dense_2_biasadd_readvariableop_resourceAdam/truediv_11:z:0%^model/dense_2/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_18ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_12AddV2Adam/ReadVariableOp_18:value:0Adam/add_12/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_18CastAdam/add_12:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_12PowAdam/Cast_19/x:output:0Adam/Cast_18:y:0*
T0*
_output_shapes
: S
Adam/Cast_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_13PowAdam/Cast_20/x:output:0Adam/Cast_18:y:0*
T0*
_output_shapes
: R
Adam/sub_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_24SubAdam/sub_24/x:output:0Adam/Pow_13:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_12SqrtAdam/sub_24:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_19ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_24MulAdam/ReadVariableOp_19:value:0Adam/Sqrt_12:y:0*
T0*
_output_shapes
: R
Adam/sub_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_25SubAdam/sub_25/x:output:0Adam/Pow_12:z:0*
T0*
_output_shapes
: ]
Adam/truediv_12RealDivAdam/mul_24:z:0Adam/sub_25:z:0*
T0*
_output_shapes
: 
Adam/sub_26/ReadVariableOpReadVariableOp#adam_sub_26_readvariableop_resource*
_output_shapes
:	�@*
dtype0t
Adam/sub_26SubIdentityN:output:6"Adam/sub_26/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@R
Adam/mul_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=e
Adam/mul_25MulAdam/sub_26:z:0Adam/mul_25/y:output:0*
T0*
_output_shapes
:	�@�
Adam/AssignAddVariableOp_12AssignAddVariableOp#adam_sub_26_readvariableop_resourceAdam/mul_25:z:0^Adam/sub_26/ReadVariableOp*
_output_shapes
 *
dtype0U
Adam/Square_6SquareIdentityN:output:6*
T0*
_output_shapes
:	�@
Adam/sub_27/ReadVariableOpReadVariableOp#adam_sub_27_readvariableop_resource*
_output_shapes
:	�@*
dtype0s
Adam/sub_27SubAdam/Square_6:y:0"Adam/sub_27/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@R
Adam/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:e
Adam/mul_26MulAdam/sub_27:z:0Adam/mul_26/y:output:0*
T0*
_output_shapes
:	�@�
Adam/AssignAddVariableOp_13AssignAddVariableOp#adam_sub_27_readvariableop_resourceAdam/mul_26:z:0^Adam/sub_27/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_20ReadVariableOp#adam_sub_26_readvariableop_resource^Adam/AssignAddVariableOp_12*
_output_shapes
:	�@*
dtype0q
Adam/mul_27MulAdam/ReadVariableOp_20:value:0Adam/truediv_12:z:0*
T0*
_output_shapes
:	�@�
Adam/Sqrt_13/ReadVariableOpReadVariableOp#adam_sub_27_readvariableop_resource^Adam/AssignAddVariableOp_13*
_output_shapes
:	�@*
dtype0c
Adam/Sqrt_13Sqrt#Adam/Sqrt_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@R
Adam/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3h
Adam/add_13AddV2Adam/Sqrt_13:y:0Adam/add_13/y:output:0*
T0*
_output_shapes
:	�@f
Adam/truediv_13RealDivAdam/mul_27:z:0Adam/add_13:z:0*
T0*
_output_shapes
:	�@�
Adam/AssignSubVariableOp_6AssignSubVariableOp,model_dense_3_matmul_readvariableop_resourceAdam/truediv_13:z:0$^model/dense_3/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_21ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_14AddV2Adam/ReadVariableOp_21:value:0Adam/add_14/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_21CastAdam/add_14:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_14PowAdam/Cast_22/x:output:0Adam/Cast_21:y:0*
T0*
_output_shapes
: S
Adam/Cast_23/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_15PowAdam/Cast_23/x:output:0Adam/Cast_21:y:0*
T0*
_output_shapes
: R
Adam/sub_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_28SubAdam/sub_28/x:output:0Adam/Pow_15:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_14SqrtAdam/sub_28:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_22ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_28MulAdam/ReadVariableOp_22:value:0Adam/Sqrt_14:y:0*
T0*
_output_shapes
: R
Adam/sub_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_29SubAdam/sub_29/x:output:0Adam/Pow_14:z:0*
T0*
_output_shapes
: ]
Adam/truediv_14RealDivAdam/mul_28:z:0Adam/sub_29:z:0*
T0*
_output_shapes
: z
Adam/sub_30/ReadVariableOpReadVariableOp#adam_sub_30_readvariableop_resource*
_output_shapes
:@*
dtype0o
Adam/sub_30SubIdentityN:output:7"Adam/sub_30/ReadVariableOp:value:0*
T0*
_output_shapes
:@R
Adam/mul_29/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=`
Adam/mul_29MulAdam/sub_30:z:0Adam/mul_29/y:output:0*
T0*
_output_shapes
:@�
Adam/AssignAddVariableOp_14AssignAddVariableOp#adam_sub_30_readvariableop_resourceAdam/mul_29:z:0^Adam/sub_30/ReadVariableOp*
_output_shapes
 *
dtype0P
Adam/Square_7SquareIdentityN:output:7*
T0*
_output_shapes
:@z
Adam/sub_31/ReadVariableOpReadVariableOp#adam_sub_31_readvariableop_resource*
_output_shapes
:@*
dtype0n
Adam/sub_31SubAdam/Square_7:y:0"Adam/sub_31/ReadVariableOp:value:0*
T0*
_output_shapes
:@R
Adam/mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:`
Adam/mul_30MulAdam/sub_31:z:0Adam/mul_30/y:output:0*
T0*
_output_shapes
:@�
Adam/AssignAddVariableOp_15AssignAddVariableOp#adam_sub_31_readvariableop_resourceAdam/mul_30:z:0^Adam/sub_31/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_23ReadVariableOp#adam_sub_30_readvariableop_resource^Adam/AssignAddVariableOp_14*
_output_shapes
:@*
dtype0l
Adam/mul_31MulAdam/ReadVariableOp_23:value:0Adam/truediv_14:z:0*
T0*
_output_shapes
:@�
Adam/Sqrt_15/ReadVariableOpReadVariableOp#adam_sub_31_readvariableop_resource^Adam/AssignAddVariableOp_15*
_output_shapes
:@*
dtype0^
Adam/Sqrt_15Sqrt#Adam/Sqrt_15/ReadVariableOp:value:0*
T0*
_output_shapes
:@R
Adam/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3c
Adam/add_15AddV2Adam/Sqrt_15:y:0Adam/add_15/y:output:0*
T0*
_output_shapes
:@a
Adam/truediv_15RealDivAdam/mul_31:z:0Adam/add_15:z:0*
T0*
_output_shapes
:@�
Adam/AssignSubVariableOp_7AssignSubVariableOp-model_dense_3_biasadd_readvariableop_resourceAdam/truediv_15:z:0%^model/dense_3/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_24ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_16/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_16AddV2Adam/ReadVariableOp_24:value:0Adam/add_16/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_24CastAdam/add_16:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_16PowAdam/Cast_25/x:output:0Adam/Cast_24:y:0*
T0*
_output_shapes
: S
Adam/Cast_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_17PowAdam/Cast_26/x:output:0Adam/Cast_24:y:0*
T0*
_output_shapes
: R
Adam/sub_32/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_32SubAdam/sub_32/x:output:0Adam/Pow_17:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_16SqrtAdam/sub_32:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_25ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_32MulAdam/ReadVariableOp_25:value:0Adam/Sqrt_16:y:0*
T0*
_output_shapes
: R
Adam/sub_33/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_33SubAdam/sub_33/x:output:0Adam/Pow_16:z:0*
T0*
_output_shapes
: ]
Adam/truediv_16RealDivAdam/mul_32:z:0Adam/sub_33:z:0*
T0*
_output_shapes
: ~
Adam/sub_34/ReadVariableOpReadVariableOp#adam_sub_34_readvariableop_resource*
_output_shapes

:@*
dtype0s
Adam/sub_34SubIdentityN:output:8"Adam/sub_34/ReadVariableOp:value:0*
T0*
_output_shapes

:@R
Adam/mul_33/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=d
Adam/mul_33MulAdam/sub_34:z:0Adam/mul_33/y:output:0*
T0*
_output_shapes

:@�
Adam/AssignAddVariableOp_16AssignAddVariableOp#adam_sub_34_readvariableop_resourceAdam/mul_33:z:0^Adam/sub_34/ReadVariableOp*
_output_shapes
 *
dtype0T
Adam/Square_8SquareIdentityN:output:8*
T0*
_output_shapes

:@~
Adam/sub_35/ReadVariableOpReadVariableOp#adam_sub_35_readvariableop_resource*
_output_shapes

:@*
dtype0r
Adam/sub_35SubAdam/Square_8:y:0"Adam/sub_35/ReadVariableOp:value:0*
T0*
_output_shapes

:@R
Adam/mul_34/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:d
Adam/mul_34MulAdam/sub_35:z:0Adam/mul_34/y:output:0*
T0*
_output_shapes

:@�
Adam/AssignAddVariableOp_17AssignAddVariableOp#adam_sub_35_readvariableop_resourceAdam/mul_34:z:0^Adam/sub_35/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_26ReadVariableOp#adam_sub_34_readvariableop_resource^Adam/AssignAddVariableOp_16*
_output_shapes

:@*
dtype0p
Adam/mul_35MulAdam/ReadVariableOp_26:value:0Adam/truediv_16:z:0*
T0*
_output_shapes

:@�
Adam/Sqrt_17/ReadVariableOpReadVariableOp#adam_sub_35_readvariableop_resource^Adam/AssignAddVariableOp_17*
_output_shapes

:@*
dtype0b
Adam/Sqrt_17Sqrt#Adam/Sqrt_17/ReadVariableOp:value:0*
T0*
_output_shapes

:@R
Adam/add_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3g
Adam/add_17AddV2Adam/Sqrt_17:y:0Adam/add_17/y:output:0*
T0*
_output_shapes

:@e
Adam/truediv_17RealDivAdam/mul_35:z:0Adam/add_17:z:0*
T0*
_output_shapes

:@�
Adam/AssignSubVariableOp_8AssignSubVariableOp,model_dense_4_matmul_readvariableop_resourceAdam/truediv_17:z:0$^model/dense_4/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0k
Adam/ReadVariableOp_27ReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	O
Adam/add_18/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rm
Adam/add_18AddV2Adam/ReadVariableOp_27:value:0Adam/add_18/y:output:0*
T0	*
_output_shapes
: U
Adam/Cast_27CastAdam/add_18:z:0*

DstT0*

SrcT0	*
_output_shapes
: S
Adam/Cast_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?^
Adam/Pow_18PowAdam/Cast_28/x:output:0Adam/Cast_27:y:0*
T0*
_output_shapes
: S
Adam/Cast_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?^
Adam/Pow_19PowAdam/Cast_29/x:output:0Adam/Cast_27:y:0*
T0*
_output_shapes
: R
Adam/sub_36/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_36SubAdam/sub_36/x:output:0Adam/Pow_19:z:0*
T0*
_output_shapes
: F
Adam/Sqrt_18SqrtAdam/sub_36:z:0*
T0*
_output_shapes
: m
Adam/ReadVariableOp_28ReadVariableOpadam_readvariableop_1_resource*
_output_shapes
: *
dtype0e
Adam/mul_36MulAdam/ReadVariableOp_28:value:0Adam/Sqrt_18:y:0*
T0*
_output_shapes
: R
Adam/sub_37/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
Adam/sub_37SubAdam/sub_37/x:output:0Adam/Pow_18:z:0*
T0*
_output_shapes
: ]
Adam/truediv_18RealDivAdam/mul_36:z:0Adam/sub_37:z:0*
T0*
_output_shapes
: z
Adam/sub_38/ReadVariableOpReadVariableOp#adam_sub_38_readvariableop_resource*
_output_shapes
:*
dtype0o
Adam/sub_38SubIdentityN:output:9"Adam/sub_38/ReadVariableOp:value:0*
T0*
_output_shapes
:R
Adam/mul_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=`
Adam/mul_37MulAdam/sub_38:z:0Adam/mul_37/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_18AssignAddVariableOp#adam_sub_38_readvariableop_resourceAdam/mul_37:z:0^Adam/sub_38/ReadVariableOp*
_output_shapes
 *
dtype0P
Adam/Square_9SquareIdentityN:output:9*
T0*
_output_shapes
:z
Adam/sub_39/ReadVariableOpReadVariableOp#adam_sub_39_readvariableop_resource*
_output_shapes
:*
dtype0n
Adam/sub_39SubAdam/Square_9:y:0"Adam/sub_39/ReadVariableOp:value:0*
T0*
_output_shapes
:R
Adam/mul_38/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:`
Adam/mul_38MulAdam/sub_39:z:0Adam/mul_38/y:output:0*
T0*
_output_shapes
:�
Adam/AssignAddVariableOp_19AssignAddVariableOp#adam_sub_39_readvariableop_resourceAdam/mul_38:z:0^Adam/sub_39/ReadVariableOp*
_output_shapes
 *
dtype0�
Adam/ReadVariableOp_29ReadVariableOp#adam_sub_38_readvariableop_resource^Adam/AssignAddVariableOp_18*
_output_shapes
:*
dtype0l
Adam/mul_39MulAdam/ReadVariableOp_29:value:0Adam/truediv_18:z:0*
T0*
_output_shapes
:�
Adam/Sqrt_19/ReadVariableOpReadVariableOp#adam_sub_39_readvariableop_resource^Adam/AssignAddVariableOp_19*
_output_shapes
:*
dtype0^
Adam/Sqrt_19Sqrt#Adam/Sqrt_19/ReadVariableOp:value:0*
T0*
_output_shapes
:R
Adam/add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3c
Adam/add_19AddV2Adam/Sqrt_19:y:0Adam/add_19/y:output:0*
T0*
_output_shapes
:a
Adam/truediv_19RealDivAdam/mul_39:z:0Adam/add_19:z:0*
T0*
_output_shapes
:�
Adam/AssignSubVariableOp_9AssignSubVariableOp-model_dense_4_biasadd_readvariableop_resourceAdam/truediv_19:z:0%^model/dense_4/BiasAdd/ReadVariableOp*
_output_shapes
 *
dtype0L

Adam/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R�
Adam/AssignAddVariableOp_20AssignAddVariableOpadam_readvariableop_resourceAdam/Const:output:0^Adam/ReadVariableOp^Adam/ReadVariableOp_12^Adam/ReadVariableOp_15^Adam/ReadVariableOp_18^Adam/ReadVariableOp_21^Adam/ReadVariableOp_24^Adam/ReadVariableOp_27^Adam/ReadVariableOp_3^Adam/ReadVariableOp_6^Adam/ReadVariableOp_9*
_output_shapes
 *
dtype0	[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������\
ArgMaxArgMaxyArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Shape_1ShapeArgMax:output:0*
T0	*
_output_shapes
::��]
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������x
ArgMax_1ArgMaxmodel/dense_4/Sigmoid:y:0ArgMax_1/dimension:output:0*
T0*#
_output_shapes
:���������`
EqualEqualArgMax:output:0ArgMax_1:output:0*
T0	*#
_output_shapes
:���������V
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*#
_output_shapes
:���������O
ConstConst*
_output_shapes
:*
dtype0*
valueB: q
Sum_2Sum
Cast_1:y:0Const:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype09
SizeSize
Cast_1:y:0*
T0*
_output_shapes
: M
Cast_2CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_2:y:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: H
Identity_10Identitydiv_no_nan:z:0*
T0*
_output_shapes
: �
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype0�
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0�
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: J
Identity_11Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: U
Identity_12IdentityIdentity_11:output:0^NoOp*
T0*
_output_shapes
: U
Identity_13IdentityIdentity_10:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^Adam/AssignAddVariableOp^Adam/AssignAddVariableOp_1^Adam/AssignAddVariableOp_10^Adam/AssignAddVariableOp_11^Adam/AssignAddVariableOp_12^Adam/AssignAddVariableOp_13^Adam/AssignAddVariableOp_14^Adam/AssignAddVariableOp_15^Adam/AssignAddVariableOp_16^Adam/AssignAddVariableOp_17^Adam/AssignAddVariableOp_18^Adam/AssignAddVariableOp_19^Adam/AssignAddVariableOp_2^Adam/AssignAddVariableOp_20^Adam/AssignAddVariableOp_3^Adam/AssignAddVariableOp_4^Adam/AssignAddVariableOp_5^Adam/AssignAddVariableOp_6^Adam/AssignAddVariableOp_7^Adam/AssignAddVariableOp_8^Adam/AssignAddVariableOp_9^Adam/AssignSubVariableOp^Adam/AssignSubVariableOp_1^Adam/AssignSubVariableOp_2^Adam/AssignSubVariableOp_3^Adam/AssignSubVariableOp_4^Adam/AssignSubVariableOp_5^Adam/AssignSubVariableOp_6^Adam/AssignSubVariableOp_7^Adam/AssignSubVariableOp_8^Adam/AssignSubVariableOp_9^Adam/ReadVariableOp^Adam/ReadVariableOp_1^Adam/ReadVariableOp_10^Adam/ReadVariableOp_11^Adam/ReadVariableOp_12^Adam/ReadVariableOp_13^Adam/ReadVariableOp_14^Adam/ReadVariableOp_15^Adam/ReadVariableOp_16^Adam/ReadVariableOp_17^Adam/ReadVariableOp_18^Adam/ReadVariableOp_19^Adam/ReadVariableOp_2^Adam/ReadVariableOp_20^Adam/ReadVariableOp_21^Adam/ReadVariableOp_22^Adam/ReadVariableOp_23^Adam/ReadVariableOp_24^Adam/ReadVariableOp_25^Adam/ReadVariableOp_26^Adam/ReadVariableOp_27^Adam/ReadVariableOp_28^Adam/ReadVariableOp_29^Adam/ReadVariableOp_3^Adam/ReadVariableOp_4^Adam/ReadVariableOp_5^Adam/ReadVariableOp_6^Adam/ReadVariableOp_7^Adam/ReadVariableOp_8^Adam/ReadVariableOp_9^Adam/Sqrt_1/ReadVariableOp^Adam/Sqrt_11/ReadVariableOp^Adam/Sqrt_13/ReadVariableOp^Adam/Sqrt_15/ReadVariableOp^Adam/Sqrt_17/ReadVariableOp^Adam/Sqrt_19/ReadVariableOp^Adam/Sqrt_3/ReadVariableOp^Adam/Sqrt_5/ReadVariableOp^Adam/Sqrt_7/ReadVariableOp^Adam/Sqrt_9/ReadVariableOp^Adam/sub_10/ReadVariableOp^Adam/sub_11/ReadVariableOp^Adam/sub_14/ReadVariableOp^Adam/sub_15/ReadVariableOp^Adam/sub_18/ReadVariableOp^Adam/sub_19/ReadVariableOp^Adam/sub_2/ReadVariableOp^Adam/sub_22/ReadVariableOp^Adam/sub_23/ReadVariableOp^Adam/sub_26/ReadVariableOp^Adam/sub_27/ReadVariableOp^Adam/sub_3/ReadVariableOp^Adam/sub_30/ReadVariableOp^Adam/sub_31/ReadVariableOp^Adam/sub_34/ReadVariableOp^Adam/sub_35/ReadVariableOp^Adam/sub_38/ReadVariableOp^Adam/sub_39/ReadVariableOp^Adam/sub_6/ReadVariableOp^Adam/sub_7/ReadVariableOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
_output_shapes
 "#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
Adam/AssignAddVariableOp_10Adam/AssignAddVariableOp_102:
Adam/AssignAddVariableOp_11Adam/AssignAddVariableOp_112:
Adam/AssignAddVariableOp_12Adam/AssignAddVariableOp_122:
Adam/AssignAddVariableOp_13Adam/AssignAddVariableOp_132:
Adam/AssignAddVariableOp_14Adam/AssignAddVariableOp_142:
Adam/AssignAddVariableOp_15Adam/AssignAddVariableOp_152:
Adam/AssignAddVariableOp_16Adam/AssignAddVariableOp_162:
Adam/AssignAddVariableOp_17Adam/AssignAddVariableOp_172:
Adam/AssignAddVariableOp_18Adam/AssignAddVariableOp_182:
Adam/AssignAddVariableOp_19Adam/AssignAddVariableOp_1928
Adam/AssignAddVariableOp_1Adam/AssignAddVariableOp_12:
Adam/AssignAddVariableOp_20Adam/AssignAddVariableOp_2028
Adam/AssignAddVariableOp_2Adam/AssignAddVariableOp_228
Adam/AssignAddVariableOp_3Adam/AssignAddVariableOp_328
Adam/AssignAddVariableOp_4Adam/AssignAddVariableOp_428
Adam/AssignAddVariableOp_5Adam/AssignAddVariableOp_528
Adam/AssignAddVariableOp_6Adam/AssignAddVariableOp_628
Adam/AssignAddVariableOp_7Adam/AssignAddVariableOp_728
Adam/AssignAddVariableOp_8Adam/AssignAddVariableOp_828
Adam/AssignAddVariableOp_9Adam/AssignAddVariableOp_924
Adam/AssignAddVariableOpAdam/AssignAddVariableOp28
Adam/AssignSubVariableOp_1Adam/AssignSubVariableOp_128
Adam/AssignSubVariableOp_2Adam/AssignSubVariableOp_228
Adam/AssignSubVariableOp_3Adam/AssignSubVariableOp_328
Adam/AssignSubVariableOp_4Adam/AssignSubVariableOp_428
Adam/AssignSubVariableOp_5Adam/AssignSubVariableOp_528
Adam/AssignSubVariableOp_6Adam/AssignSubVariableOp_628
Adam/AssignSubVariableOp_7Adam/AssignSubVariableOp_728
Adam/AssignSubVariableOp_8Adam/AssignSubVariableOp_828
Adam/AssignSubVariableOp_9Adam/AssignSubVariableOp_924
Adam/AssignSubVariableOpAdam/AssignSubVariableOp20
Adam/ReadVariableOp_10Adam/ReadVariableOp_1020
Adam/ReadVariableOp_11Adam/ReadVariableOp_1120
Adam/ReadVariableOp_12Adam/ReadVariableOp_1220
Adam/ReadVariableOp_13Adam/ReadVariableOp_1320
Adam/ReadVariableOp_14Adam/ReadVariableOp_1420
Adam/ReadVariableOp_15Adam/ReadVariableOp_1520
Adam/ReadVariableOp_16Adam/ReadVariableOp_1620
Adam/ReadVariableOp_17Adam/ReadVariableOp_1720
Adam/ReadVariableOp_18Adam/ReadVariableOp_1820
Adam/ReadVariableOp_19Adam/ReadVariableOp_192.
Adam/ReadVariableOp_1Adam/ReadVariableOp_120
Adam/ReadVariableOp_20Adam/ReadVariableOp_2020
Adam/ReadVariableOp_21Adam/ReadVariableOp_2120
Adam/ReadVariableOp_22Adam/ReadVariableOp_2220
Adam/ReadVariableOp_23Adam/ReadVariableOp_2320
Adam/ReadVariableOp_24Adam/ReadVariableOp_2420
Adam/ReadVariableOp_25Adam/ReadVariableOp_2520
Adam/ReadVariableOp_26Adam/ReadVariableOp_2620
Adam/ReadVariableOp_27Adam/ReadVariableOp_2720
Adam/ReadVariableOp_28Adam/ReadVariableOp_2820
Adam/ReadVariableOp_29Adam/ReadVariableOp_292.
Adam/ReadVariableOp_2Adam/ReadVariableOp_22.
Adam/ReadVariableOp_3Adam/ReadVariableOp_32.
Adam/ReadVariableOp_4Adam/ReadVariableOp_42.
Adam/ReadVariableOp_5Adam/ReadVariableOp_52.
Adam/ReadVariableOp_6Adam/ReadVariableOp_62.
Adam/ReadVariableOp_7Adam/ReadVariableOp_72.
Adam/ReadVariableOp_8Adam/ReadVariableOp_82.
Adam/ReadVariableOp_9Adam/ReadVariableOp_92*
Adam/ReadVariableOpAdam/ReadVariableOp28
Adam/Sqrt_1/ReadVariableOpAdam/Sqrt_1/ReadVariableOp2:
Adam/Sqrt_11/ReadVariableOpAdam/Sqrt_11/ReadVariableOp2:
Adam/Sqrt_13/ReadVariableOpAdam/Sqrt_13/ReadVariableOp2:
Adam/Sqrt_15/ReadVariableOpAdam/Sqrt_15/ReadVariableOp2:
Adam/Sqrt_17/ReadVariableOpAdam/Sqrt_17/ReadVariableOp2:
Adam/Sqrt_19/ReadVariableOpAdam/Sqrt_19/ReadVariableOp28
Adam/Sqrt_3/ReadVariableOpAdam/Sqrt_3/ReadVariableOp28
Adam/Sqrt_5/ReadVariableOpAdam/Sqrt_5/ReadVariableOp28
Adam/Sqrt_7/ReadVariableOpAdam/Sqrt_7/ReadVariableOp28
Adam/Sqrt_9/ReadVariableOpAdam/Sqrt_9/ReadVariableOp28
Adam/sub_10/ReadVariableOpAdam/sub_10/ReadVariableOp28
Adam/sub_11/ReadVariableOpAdam/sub_11/ReadVariableOp28
Adam/sub_14/ReadVariableOpAdam/sub_14/ReadVariableOp28
Adam/sub_15/ReadVariableOpAdam/sub_15/ReadVariableOp28
Adam/sub_18/ReadVariableOpAdam/sub_18/ReadVariableOp28
Adam/sub_19/ReadVariableOpAdam/sub_19/ReadVariableOp26
Adam/sub_2/ReadVariableOpAdam/sub_2/ReadVariableOp28
Adam/sub_22/ReadVariableOpAdam/sub_22/ReadVariableOp28
Adam/sub_23/ReadVariableOpAdam/sub_23/ReadVariableOp28
Adam/sub_26/ReadVariableOpAdam/sub_26/ReadVariableOp28
Adam/sub_27/ReadVariableOpAdam/sub_27/ReadVariableOp26
Adam/sub_3/ReadVariableOpAdam/sub_3/ReadVariableOp28
Adam/sub_30/ReadVariableOpAdam/sub_30/ReadVariableOp28
Adam/sub_31/ReadVariableOpAdam/sub_31/ReadVariableOp28
Adam/sub_34/ReadVariableOpAdam/sub_34/ReadVariableOp28
Adam/sub_35/ReadVariableOpAdam/sub_35/ReadVariableOp28
Adam/sub_38/ReadVariableOpAdam/sub_38/ReadVariableOp28
Adam/sub_39/ReadVariableOpAdam/sub_39/ReadVariableOp26
Adam/sub_6/ReadVariableOpAdam/sub_6/ReadVariableOp26
Adam/sub_7/ReadVariableOpAdam/sub_7/ReadVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32*
AssignAddVariableOpAssignAddVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:JF
'
_output_shapes
:���������

_user_specified_namey:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
$__inference_dense_layer_call_fn_3154

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2861o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name3150:$ 

_user_specified_name3148:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_3028
input_1
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$
 

_user_specified_name3024:$	 

_user_specified_name3022:$ 

_user_specified_name3020:$ 

_user_specified_name3018:$ 

_user_specified_name3016:$ 

_user_specified_name3014:$ 

_user_specified_name3012:$ 

_user_specified_name3010:$ 

_user_specified_name3008:$ 

_user_specified_name3006:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
A__inference_dense_3_layer_call_and_return_conditional_losses_3279

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_1_layer_call_and_return_conditional_losses_2877

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_3306

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
A__inference_dense_1_layer_call_and_return_conditional_losses_3185

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
B
&__inference_dropout_layer_call_fn_3195

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_3076a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
?__inference_model_layer_call_and_return_conditional_losses_3003
input_1

dense_2974:

dense_2976:
dense_1_2979:	�
dense_1_2981:	� 
dense_2_2985:
��
dense_2_2987:	�
dense_3_2991:	�@
dense_3_2993:@
dense_4_2997:@
dense_4_2999:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1
dense_2974
dense_2976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2861�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2979dense_1_2981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2877�
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2894�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_2985dense_2_2987*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2906�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_2923�
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_2991dense_3_2993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2935�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2952�
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_2997dense_4_2999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2964w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:$
 

_user_specified_name2999:$	 

_user_specified_name2997:$ 

_user_specified_name2993:$ 

_user_specified_name2991:$ 

_user_specified_name2987:$ 

_user_specified_name2985:$ 

_user_specified_name2981:$ 

_user_specified_name2979:$ 

_user_specified_name2976:$ 

_user_specified_name2974:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
a
(__inference_dropout_2_layer_call_fn_3284

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_dense_4_layer_call_fn_3315

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name3311:$ 

_user_specified_name3309:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�+
�
__inference_parameters_236.
read_readvariableop_resource:,
read_1_readvariableop_resource:1
read_2_readvariableop_resource:	�-
read_3_readvariableop_resource:	�2
read_4_readvariableop_resource:
��-
read_5_readvariableop_resource:	�1
read_6_readvariableop_resource:	�@,
read_7_readvariableop_resource:@0
read_8_readvariableop_resource:@,
read_9_readvariableop_resource:
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOp�Read_4/ReadVariableOp�Read_5/ReadVariableOp�Read_6/ReadVariableOp�Read_7/ReadVariableOp�Read_8/ReadVariableOp�Read_9/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:p
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:*
dtype0Z

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�q
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes	
:�*
dtype0[

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�v
Read_4/ReadVariableOpReadVariableOpread_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0`

Identity_4IdentityRead_4/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��q
Read_5/ReadVariableOpReadVariableOpread_5_readvariableop_resource*
_output_shapes	
:�*
dtype0[

Identity_5IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes	
:�u
Read_6/ReadVariableOpReadVariableOpread_6_readvariableop_resource*
_output_shapes
:	�@*
dtype0_

Identity_6IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@p
Read_7/ReadVariableOpReadVariableOpread_7_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_7IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@t
Read_8/ReadVariableOpReadVariableOpread_8_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_8IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_9/ReadVariableOpReadVariableOpread_9_readvariableop_resource*
_output_shapes
:*
dtype0Z

Identity_9IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:Z
Identity_10IdentityIdentity:output:0^NoOp*
T0*
_output_shapes

:X
Identity_11IdentityIdentity_1:output:0^NoOp*
T0*
_output_shapes
:]
Identity_12IdentityIdentity_2:output:0^NoOp*
T0*
_output_shapes
:	�Y
Identity_13IdentityIdentity_3:output:0^NoOp*
T0*
_output_shapes	
:�^
Identity_14IdentityIdentity_4:output:0^NoOp*
T0* 
_output_shapes
:
��Y
Identity_15IdentityIdentity_5:output:0^NoOp*
T0*
_output_shapes	
:�]
Identity_16IdentityIdentity_6:output:0^NoOp*
T0*
_output_shapes
:	�@X
Identity_17IdentityIdentity_7:output:0^NoOp*
T0*
_output_shapes
:@\
Identity_18IdentityIdentity_8:output:0^NoOp*
T0*
_output_shapes

:@X
Identity_19IdentityIdentity_9:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp^Read_4/ReadVariableOp^Read_5/ReadVariableOp^Read_6/ReadVariableOp^Read_7/ReadVariableOp^Read_8/ReadVariableOp^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp2.
Read_4/ReadVariableOpRead_4/ReadVariableOp2.
Read_5/ReadVariableOpRead_5/ReadVariableOp2.
Read_6/ReadVariableOpRead_6/ReadVariableOp2.
Read_7/ReadVariableOpRead_7/ReadVariableOp2.
Read_8/ReadVariableOpRead_8/ReadVariableOp2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource8
!__inference_internal_grad_fn_3498CustomGradient-23888
!__inference_internal_grad_fn_3561CustomGradient-1317"�L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
infer
%
x 
	infer_x:0���������:
logits0
StatefulPartitionedCall:0���������tensorflow/serving/predict*�

parameters�/
a0)
StatefulPartitionedCall_1:0+
a1%
StatefulPartitionedCall_1:10
a2*
StatefulPartitionedCall_1:2	�,
a3&
StatefulPartitionedCall_1:3�1
a4+
StatefulPartitionedCall_1:4
��,
a5&
StatefulPartitionedCall_1:5�0
a6*
StatefulPartitionedCall_1:6	�@+
a7%
StatefulPartitionedCall_1:7@/
a8)
StatefulPartitionedCall_1:8@+
a9%
StatefulPartitionedCall_1:9tensorflow/serving/predict*�
restore�
 
a0
restore_a0:0

a1
restore_a1:0
!
a2
restore_a2:0	�

a3
restore_a3:0�
"
a4
restore_a4:0
��

a5
restore_a5:0�
!
a6
restore_a6:0	�@

a7
restore_a7:0@
 
a8
restore_a8:0@

a9
restore_a9:0/
a0)
StatefulPartitionedCall_2:0+
a1%
StatefulPartitionedCall_2:10
a2*
StatefulPartitionedCall_2:2	�,
a3&
StatefulPartitionedCall_2:3�1
a4+
StatefulPartitionedCall_2:4
��,
a5&
StatefulPartitionedCall_2:5�0
a6*
StatefulPartitionedCall_2:6	�@+
a7%
StatefulPartitionedCall_2:7@/
a8)
StatefulPartitionedCall_2:8@+
a9%
StatefulPartitionedCall_2:9tensorflow/serving/predict*�
train�
%
x 
	train_x:0���������
%
y 
	train_y:0���������-
accuracy!
StatefulPartitionedCall_3:0 )
loss!
StatefulPartitionedCall_3:1 tensorflow/serving/predict:��
l
	model
	infer

parameters
restore
	train

signatures"
_generic_user_object
�
layer-0
layer_with_weights-0
layer-1
	layer_with_weights-1
	layer-2

layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer_with_weights-4
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer"
_tf_keras_network
�
trace_02�
__inference_infer_2043�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
����������ztrace_0
�
trace_02�
__inference_parameters_2085�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� ztrace_0
�
trace_02�
__inference_restore_2137�
���
FullArgSpec
args� 
varargs
 
varkwj
parameters
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
__inference_train_2785�
���
FullArgSpec
args�

jx
jy
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
����������
����������ztrace_0
J
	train
	infer

parameters
restore"
signature_map
"
_tf_keras_input_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
E_random_generator"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias"
_tf_keras_layer
f
&0
'1
.2
/3
=4
>5
L6
M7
[8
\9"
trackable_list_wrapper
f
&0
'1
.2
/3
=4
>5
L6
M7
[8
\9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
btrace_0
ctrace_12�
$__inference_model_layer_call_fn_3028
$__inference_model_layer_call_fn_3053�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0zctrace_1
�
dtrace_0
etrace_12�
?__inference_model_layer_call_and_return_conditional_losses_2971
?__inference_model_layer_call_and_return_conditional_losses_3003�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1
�B�
__inference__wrapped_model_2848input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
f
_variables
g_iterations
h_learning_rate
i_index_dict
j
_momentums
k_velocities
l_update_step_xla"
experimentalOptimizer
�B�
__inference_infer_2043x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_parameters_2085"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_restore_2137a0a1a2a3a4a5a6a7a8a9
"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 M

kwonlyargs?�<
ja0
ja1
ja2
ja3
ja4
ja5
ja6
ja7
ja8
ja9
kwonlydefaults
 
annotations� *
 
�B�
__inference_train_2785xy"�
���
FullArgSpec
args�

jx
jy
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_1858xy"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jx
jy
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_1884x"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
jx
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_1927"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
"__inference_signature_wrapper_1980a0a1a2a3a4a5a6a7a8a9"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 M

kwonlyargs?�<
ja0
ja1
ja2
ja3
ja4
ja5
ja6
ja7
ja8
ja9
kwonlydefaults
 
annotations� *
 
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_02�
$__inference_dense_layer_call_fn_3154�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
�
strace_02�
?__inference_dense_layer_call_and_return_conditional_losses_3165�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0
:2dense/kernel
:2
dense/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
ytrace_02�
&__inference_dense_1_layer_call_fn_3174�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
�
ztrace_02�
A__inference_dense_1_layer_call_and_return_conditional_losses_3185�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zztrace_0
!:	�2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_dropout_layer_call_fn_3190
&__inference_dropout_layer_call_fn_3195�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_dropout_layer_call_and_return_conditional_losses_3207
A__inference_dropout_layer_call_and_return_conditional_losses_3212�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_2_layer_call_fn_3221�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_2_layer_call_and_return_conditional_losses_3232�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_2/kernel
:�2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_1_layer_call_fn_3237
(__inference_dropout_1_layer_call_fn_3242�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_1_layer_call_and_return_conditional_losses_3254
C__inference_dropout_1_layer_call_and_return_conditional_losses_3259�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_3_layer_call_fn_3268�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_3_layer_call_and_return_conditional_losses_3279�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�@2dense_3/kernel
:@2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_2_layer_call_fn_3284
(__inference_dropout_2_layer_call_fn_3289�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_2_layer_call_and_return_conditional_losses_3301
C__inference_dropout_2_layer_call_and_return_conditional_losses_3306�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_4_layer_call_fn_3315�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_4_layer_call_and_return_conditional_losses_3326�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :@2dense_4/kernel
:2dense_4/bias
 "
trackable_list_wrapper
_
0
1
	2

3
4
5
6
7
8"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_3028input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3053input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_2971input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_3003input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
g0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_dense_layer_call_fn_3154inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dense_layer_call_and_return_conditional_losses_3165inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_1_layer_call_fn_3174inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_1_layer_call_and_return_conditional_losses_3185inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dropout_layer_call_fn_3190inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_dropout_layer_call_fn_3195inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dropout_layer_call_and_return_conditional_losses_3207inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dropout_layer_call_and_return_conditional_losses_3212inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_2_layer_call_fn_3221inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_2_layer_call_and_return_conditional_losses_3232inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dropout_1_layer_call_fn_3237inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_1_layer_call_fn_3242inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_1_layer_call_and_return_conditional_losses_3254inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_1_layer_call_and_return_conditional_losses_3259inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_3_layer_call_fn_3268inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_3_layer_call_and_return_conditional_losses_3279inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dropout_2_layer_call_fn_3284inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_2_layer_call_fn_3289inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_2_layer_call_and_return_conditional_losses_3301inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_2_layer_call_and_return_conditional_losses_3306inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_4_layer_call_fn_3315inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_4_layer_call_and_return_conditional_losses_3326inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
#:!2Adam/m/dense/kernel
#:!2Adam/v/dense/kernel
:2Adam/m/dense/bias
:2Adam/v/dense/bias
&:$	�2Adam/m/dense_1/kernel
&:$	�2Adam/v/dense_1/kernel
 :�2Adam/m/dense_1/bias
 :�2Adam/v/dense_1/bias
':%
��2Adam/m/dense_2/kernel
':%
��2Adam/v/dense_2/kernel
 :�2Adam/m/dense_2/bias
 :�2Adam/v/dense_2/bias
&:$	�@2Adam/m/dense_3/kernel
&:$	�@2Adam/v/dense_3/kernel
:@2Adam/m/dense_3/bias
:@2Adam/v/dense_3/bias
%:#@2Adam/m/dense_4/kernel
%:#@2Adam/v/dense_4/kernel
:2Adam/m/dense_4/bias
:2Adam/v/dense_4/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
__inference__wrapped_model_2848q
&'./=>LM[\0�-
&�#
!�
input_1���������
� "1�.
,
dense_4!�
dense_4����������
A__inference_dense_1_layer_call_and_return_conditional_losses_3185d.//�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_1_layer_call_fn_3174Y.//�,
%�"
 �
inputs���������
� ""�
unknown�����������
A__inference_dense_2_layer_call_and_return_conditional_losses_3232e=>0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_2_layer_call_fn_3221Z=>0�-
&�#
!�
inputs����������
� ""�
unknown�����������
A__inference_dense_3_layer_call_and_return_conditional_losses_3279dLM0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
&__inference_dense_3_layer_call_fn_3268YLM0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
A__inference_dense_4_layer_call_and_return_conditional_losses_3326c[\/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
&__inference_dense_4_layer_call_fn_3315X[\/�,
%�"
 �
inputs���������@
� "!�
unknown����������
?__inference_dense_layer_call_and_return_conditional_losses_3165c&'/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
$__inference_dense_layer_call_fn_3154X&'/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_dropout_1_layer_call_and_return_conditional_losses_3254e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
C__inference_dropout_1_layer_call_and_return_conditional_losses_3259e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
(__inference_dropout_1_layer_call_fn_3237Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
(__inference_dropout_1_layer_call_fn_3242Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
C__inference_dropout_2_layer_call_and_return_conditional_losses_3301c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
C__inference_dropout_2_layer_call_and_return_conditional_losses_3306c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
(__inference_dropout_2_layer_call_fn_3284X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
(__inference_dropout_2_layer_call_fn_3289X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
A__inference_dropout_layer_call_and_return_conditional_losses_3207e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
A__inference_dropout_layer_call_and_return_conditional_losses_3212e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
&__inference_dropout_layer_call_fn_3190Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
&__inference_dropout_layer_call_fn_3195Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
__inference_infer_2043i
&'./=>LM[\*�'
 �
�
x���������
� "/�,
*
logits �
logits����������
!__inference_internal_grad_fn_3498����
���

 
�
result_grads_0
�
result_grads_1
 �
result_grads_2	�
�
result_grads_3�
!�
result_grads_4
��
�
result_grads_5�
 �
result_grads_6	�@
�
result_grads_7@
�
result_grads_8@
�
result_grads_9
 �
result_grads_10
�
result_grads_11
!�
result_grads_12	�
�
result_grads_13�
"�
result_grads_14
��
�
result_grads_15�
!�
result_grads_16	�@
�
result_grads_17@
 �
result_grads_18@
�
result_grads_19
� "���

 

 

 

 

 

 

 

 

 

 
�
	tensor_10
�
	tensor_11
�
	tensor_12	�
�
	tensor_13�
�
	tensor_14
��
�
	tensor_15�
�
	tensor_16	�@
�
	tensor_17@
�
	tensor_18@
�
	tensor_19�
!__inference_internal_grad_fn_3561����
���

 
�
result_grads_0
�
result_grads_1
 �
result_grads_2	�
�
result_grads_3�
!�
result_grads_4
��
�
result_grads_5�
 �
result_grads_6	�@
�
result_grads_7@
�
result_grads_8@
�
result_grads_9
 �
result_grads_10
�
result_grads_11
!�
result_grads_12	�
�
result_grads_13�
"�
result_grads_14
��
�
result_grads_15�
!�
result_grads_16	�@
�
result_grads_17@
 �
result_grads_18@
�
result_grads_19
� "���

 

 

 

 

 

 

 

 

 

 
�
	tensor_10
�
	tensor_11
�
	tensor_12	�
�
	tensor_13�
�
	tensor_14
��
�
	tensor_15�
�
	tensor_16	�@
�
	tensor_17@
�
	tensor_18@
�
	tensor_19�
?__inference_model_layer_call_and_return_conditional_losses_2971t
&'./=>LM[\8�5
.�+
!�
input_1���������
p

 
� ",�)
"�
tensor_0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_3003t
&'./=>LM[\8�5
.�+
!�
input_1���������
p 

 
� ",�)
"�
tensor_0���������
� �
$__inference_model_layer_call_fn_3028i
&'./=>LM[\8�5
.�+
!�
input_1���������
p

 
� "!�
unknown����������
$__inference_model_layer_call_fn_3053i
&'./=>LM[\8�5
.�+
!�
input_1���������
p 

 
� "!�
unknown����������
__inference_parameters_2085�
&'./=>LM[\�

� 
� "���

a0�
a0

a1�
a1

a2�
a2	�

a3�
a3�

a4�
a4
��

a5�
a5�

a6�
a6	�@

a7�
a7@

a8�
a8@

a9�
a9�
__inference_restore_2137�
&'./=>LM[\���
� 
���

a0�
a0

a1�
a1

a2�
a2	�

a3�
a3�

a4�
a4
��

a5�
a5�

a6�
a6	�@

a7�
a7@

a8�
a8@

a9�
a9"���

a0�
a0

a1�
a1

a2�
a2	�

a3�
a3�

a4�
a4
��

a5�
a5�

a6�
a6	�@

a7�
a7@

a8�
a8@

a9�
a9�
"__inference_signature_wrapper_1858�<&'./=>LM[\��gh����������������������Q�N
� 
G�D
 
x�
x���������
 
y�
y���������"9�6

accuracy�
accuracy 

loss�

loss �
"__inference_signature_wrapper_1884n
&'./=>LM[\/�,
� 
%�"
 
x�
x���������"/�,
*
logits �
logits����������
"__inference_signature_wrapper_1927�
&'./=>LM[\�

� 
� "���

a0�
a0

a1�
a1

a2�
a2	�

a3�
a3�

a4�
a4
��

a5�
a5�

a6�
a6	�@

a7�
a7@

a8�
a8@

a9�
a9�
"__inference_signature_wrapper_1980�
&'./=>LM[\���
� 
���

a0�
a0

a1�
a1

a2�
a2	�

a3�
a3�

a4�
a4
��

a5�
a5�

a6�
a6	�@

a7�
a7@

a8�
a8@

a9�
a9"���

a0�
a0

a1�
a1

a2�
a2	�

a3�
a3�

a4�
a4
��

a5�
a5�

a6�
a6	�@

a7�
a7@

a8�
a8@

a9�
a9�
__inference_train_2785�<&'./=>LM[\��gh����������������������G�D
=�:
�
x���������
�
y���������
� "9�6

accuracy�
accuracy 

loss�

loss 