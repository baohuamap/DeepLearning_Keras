Ù	
Ô
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68È´

conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
:@*
dtype0

conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:@*
dtype0
|
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@* 
shared_namedense_32/kernel
u
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel* 
_output_shapes
:
@*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:@*
dtype0
z
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

:@*
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_32/kernel/m

+Adam/conv2d_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_32/bias/m
{
)Adam/conv2d_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_33/kernel/m

+Adam/conv2d_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_33/bias/m
{
)Adam/conv2d_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*'
shared_nameAdam/dense_32/kernel/m

*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m* 
_output_shapes
:
@*
dtype0

Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_33/kernel/m

*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/m
y
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_32/kernel/v

+Adam/conv2d_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_32/bias/v
{
)Adam/conv2d_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_33/kernel/v

+Adam/conv2d_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_33/bias/v
{
)Adam/conv2d_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*'
shared_nameAdam/dense_32/kernel/v

*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v* 
_output_shapes
:
@*
dtype0

Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_33/kernel/v

*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/v
y
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ÀR
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ûQ
valueñQBîQ BçQ
Ä
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 

#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
¦

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*

1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 

7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
¦

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses*

K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
¦

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*

Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
ä
_iter

`beta_1

abeta_2
	bdecay
clearning_ratem¬m­)m®*m¯Cm°Dm±Qm²Rm³v´vµ)v¶*v·Cv¸Dv¹QvºRv»*
<
0
1
)2
*3
C4
D5
Q6
R7*
<
0
1
)2
*3
C4
D5
Q6
R7*
* 
°
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

iserving_default* 
`Z
VARIABLE_VALUEconv2d_32/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_32/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_33/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_33/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_32/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_33/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

¡0
¢1*
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

£total

¤count
¥	variables
¦	keras_api*
M

§total

¨count
©
_fn_kwargs
ª	variables
«	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

£0
¤1*

¥	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

§0
¨1*

ª	variables*
}
VARIABLE_VALUEAdam/conv2d_32/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_32/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_33/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_33/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_32/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_32/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_33/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_33/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_conv2d_32_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿFF
Î
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_32_inputconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_141251
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_32/kernel/m/Read/ReadVariableOp)Adam/conv2d_32/bias/m/Read/ReadVariableOp+Adam/conv2d_33/kernel/m/Read/ReadVariableOp)Adam/conv2d_33/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp+Adam/conv2d_32/kernel/v/Read/ReadVariableOp)Adam/conv2d_32/bias/v/Read/ReadVariableOp+Adam/conv2d_33/kernel/v/Read/ReadVariableOp)Adam/conv2d_33/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_141520

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_32/kernel/mAdam/conv2d_32/bias/mAdam/conv2d_33/kernel/mAdam/conv2d_33/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/conv2d_32/kernel/vAdam/conv2d_32/bias/vAdam/conv2d_33/kernel/vAdam/conv2d_33/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/vAdam/dense_33/kernel/vAdam/dense_33/bias/v*-
Tin&
$2"*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_141629ÿ
Â

)__inference_dense_33_layer_call_fn_141378

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_140849o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç	
õ
D__inference_dense_33_layer_call_and_return_conditional_losses_141388

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ê
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_140814

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
e
I__inference_activation_58_layer_call_and_return_conditional_losses_141369

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾+

I__inference_sequential_19_layer_call_and_return_conditional_losses_141108
conv2d_32_input*
conv2d_32_141080:@
conv2d_32_141082:@*
conv2d_33_141087:@@
conv2d_33_141089:@#
dense_32_141095:
@
dense_32_141097:@!
dense_33_141101:@
dense_33_141103:
identity¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputconv2d_32_141080conv2d_32_141082*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_140770î
activation_56/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_56_layer_call_and_return_conditional_losses_140781ð
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140738
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0conv2d_33_141087conv2d_33_141089*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_140794î
activation_57/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_57_layer_call_and_return_conditional_losses_140805ð
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_140750á
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_140814
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_141095dense_32_141097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_140826å
activation_58/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_58_layer_call_and_return_conditional_losses_140837
 dense_33/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0dense_33_141101dense_33_141103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_140849å
activation_59/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_59_layer_call_and_return_conditional_losses_140860u
IdentityIdentity&activation_59/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
)
_user_specified_nameconv2d_32_input
í
e
I__inference_activation_56_layer_call_and_return_conditional_losses_140781

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿDD@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@
 
_user_specified_nameinputs
£+

I__inference_sequential_19_layer_call_and_return_conditional_losses_140863

inputs*
conv2d_32_140771:@
conv2d_32_140773:@*
conv2d_33_140795:@@
conv2d_33_140797:@#
dense_32_140827:
@
dense_32_140829:@!
dense_33_140850:@
dense_33_140852:
identity¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCallü
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32_140771conv2d_32_140773*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_140770î
activation_56/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_56_layer_call_and_return_conditional_losses_140781ð
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140738
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0conv2d_33_140795conv2d_33_140797*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_140794î
activation_57/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_57_layer_call_and_return_conditional_losses_140805ð
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_140750á
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_140814
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_140827dense_32_140829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_140826å
activation_58/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_58_layer_call_and_return_conditional_losses_140837
 dense_33/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0dense_33_140850dense_33_140852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_140849å
activation_59/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_59_layer_call_and_return_conditional_losses_140860u
IdentityIdentity&activation_59/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs
Ï	
÷
D__inference_dense_32_layer_call_and_return_conditional_losses_140826

inputs2
matmul_readvariableop_resource:
@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_33_layer_call_and_return_conditional_losses_140794

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ""@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_140750

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê	
Ï
.__inference_sequential_19_layer_call_fn_141135

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:
@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_140863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs
ê	
Ï
.__inference_sequential_19_layer_call_fn_141156

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:
@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_141006o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs
¾+

I__inference_sequential_19_layer_call_and_return_conditional_losses_141077
conv2d_32_input*
conv2d_32_141049:@
conv2d_32_141051:@*
conv2d_33_141056:@@
conv2d_33_141058:@#
dense_32_141064:
@
dense_32_141066:@!
dense_33_141070:@
dense_33_141072:
identity¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputconv2d_32_141049conv2d_32_141051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_140770î
activation_56/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_56_layer_call_and_return_conditional_losses_140781ð
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140738
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0conv2d_33_141056conv2d_33_141058*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_140794î
activation_57/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_57_layer_call_and_return_conditional_losses_140805ð
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_140750á
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_140814
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_141064dense_32_141066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_140826å
activation_58/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_58_layer_call_and_return_conditional_losses_140837
 dense_33/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0dense_33_141070dense_33_141072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_140849å
activation_59/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_59_layer_call_and_return_conditional_losses_140860u
IdentityIdentity&activation_59/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
)
_user_specified_nameconv2d_32_input
Ï	
÷
D__inference_dense_32_layer_call_and_return_conditional_losses_141359

inputs2
matmul_readvariableop_resource:
@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
J
.__inference_activation_59_layer_call_fn_141393

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_59_layer_call_and_return_conditional_losses_140860`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
M
1__inference_max_pooling2d_32_layer_call_fn_141285

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140738
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

*__inference_conv2d_32_layer_call_fn_141260

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_140770w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿFF: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs
Ç
J
.__inference_activation_57_layer_call_fn_141314

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_57_layer_call_and_return_conditional_losses_140805h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  @:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @
 
_user_specified_nameinputs
Ê
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_141340

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ì
e
I__inference_activation_59_layer_call_and_return_conditional_losses_141398

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£+
Ú
I__inference_sequential_19_layer_call_and_return_conditional_losses_141192

inputsB
(conv2d_32_conv2d_readvariableop_resource:@7
)conv2d_32_biasadd_readvariableop_resource:@B
(conv2d_33_conv2d_readvariableop_resource:@@7
)conv2d_33_biasadd_readvariableop_resource:@;
'dense_32_matmul_readvariableop_resource:
@6
(dense_32_biasadd_readvariableop_resource:@9
'dense_33_matmul_readvariableop_resource:@6
(dense_33_biasadd_readvariableop_resource:
identity¢ conv2d_32/BiasAdd/ReadVariableOp¢conv2d_32/Conv2D/ReadVariableOp¢ conv2d_33/BiasAdd/ReadVariableOp¢conv2d_33/Conv2D/ReadVariableOp¢dense_32/BiasAdd/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/BiasAdd/ReadVariableOp¢dense_33/MatMul/ReadVariableOp
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0®
conv2d_32/Conv2DConv2Dinputs'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*
paddingVALID*
strides

 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@p
activation_56/ReluReluconv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@²
max_pooling2d_32/MaxPoolMaxPool activation_56/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@*
ksize
*
paddingVALID*
strides

conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0É
conv2d_33/Conv2DConv2D!max_pooling2d_32/MaxPool:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingVALID*
strides

 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @p
activation_57/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @²
max_pooling2d_33/MaxPoolMaxPool activation_57/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
a
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  
flatten_16/ReshapeReshape!max_pooling2d_33/MaxPool:output:0flatten_16/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0
dense_32/MatMulMatMulflatten_16/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
activation_58/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_33/MatMulMatMul activation_58/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
activation_59/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityactivation_59/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
NoOpNoOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_141329

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Ø
.__inference_sequential_19_layer_call_fn_141046
conv2d_32_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:
@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_141006o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
)
_user_specified_nameconv2d_32_input
Ì
e
I__inference_activation_59_layer_call_and_return_conditional_losses_140860

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_141290

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
M
1__inference_max_pooling2d_33_layer_call_fn_141324

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_140750
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Ø
.__inference_sequential_19_layer_call_fn_140882
conv2d_32_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:
@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_140863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
)
_user_specified_nameconv2d_32_input
Ç	
õ
D__inference_dense_33_layer_call_and_return_conditional_losses_140849

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
£+
Ú
I__inference_sequential_19_layer_call_and_return_conditional_losses_141228

inputsB
(conv2d_32_conv2d_readvariableop_resource:@7
)conv2d_32_biasadd_readvariableop_resource:@B
(conv2d_33_conv2d_readvariableop_resource:@@7
)conv2d_33_biasadd_readvariableop_resource:@;
'dense_32_matmul_readvariableop_resource:
@6
(dense_32_biasadd_readvariableop_resource:@9
'dense_33_matmul_readvariableop_resource:@6
(dense_33_biasadd_readvariableop_resource:
identity¢ conv2d_32/BiasAdd/ReadVariableOp¢conv2d_32/Conv2D/ReadVariableOp¢ conv2d_33/BiasAdd/ReadVariableOp¢conv2d_33/Conv2D/ReadVariableOp¢dense_32/BiasAdd/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/BiasAdd/ReadVariableOp¢dense_33/MatMul/ReadVariableOp
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0®
conv2d_32/Conv2DConv2Dinputs'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*
paddingVALID*
strides

 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@p
activation_56/ReluReluconv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@²
max_pooling2d_32/MaxPoolMaxPool activation_56/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@*
ksize
*
paddingVALID*
strides

conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0É
conv2d_33/Conv2DConv2D!max_pooling2d_32/MaxPool:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingVALID*
strides

 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @p
activation_57/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @²
max_pooling2d_33/MaxPoolMaxPool activation_57/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
a
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  
flatten_16/ReshapeReshape!max_pooling2d_33/MaxPool:output:0flatten_16/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0
dense_32/MatMulMatMulflatten_16/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
activation_58/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_33/MatMulMatMul activation_58/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
activation_59/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityactivation_59/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
NoOpNoOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs
×

"__inference__traced_restore_141629
file_prefix;
!assignvariableop_conv2d_32_kernel:@/
!assignvariableop_1_conv2d_32_bias:@=
#assignvariableop_2_conv2d_33_kernel:@@/
!assignvariableop_3_conv2d_33_bias:@6
"assignvariableop_4_dense_32_kernel:
@.
 assignvariableop_5_dense_32_bias:@4
"assignvariableop_6_dense_33_kernel:@.
 assignvariableop_7_dense_33_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: E
+assignvariableop_17_adam_conv2d_32_kernel_m:@7
)assignvariableop_18_adam_conv2d_32_bias_m:@E
+assignvariableop_19_adam_conv2d_33_kernel_m:@@7
)assignvariableop_20_adam_conv2d_33_bias_m:@>
*assignvariableop_21_adam_dense_32_kernel_m:
@6
(assignvariableop_22_adam_dense_32_bias_m:@<
*assignvariableop_23_adam_dense_33_kernel_m:@6
(assignvariableop_24_adam_dense_33_bias_m:E
+assignvariableop_25_adam_conv2d_32_kernel_v:@7
)assignvariableop_26_adam_conv2d_32_bias_v:@E
+assignvariableop_27_adam_conv2d_33_kernel_v:@@7
)assignvariableop_28_adam_conv2d_33_bias_v:@>
*assignvariableop_29_adam_dense_32_kernel_v:
@6
(assignvariableop_30_adam_dense_32_bias_v:@<
*assignvariableop_31_adam_dense_33_kernel_v:@6
(assignvariableop_32_adam_dense_33_bias_v:
identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9²
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_33_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_33_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_32_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_32_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_33_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_33_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_32_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_32_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_33_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_33_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_32_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_32_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_33_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_33_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_32_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_32_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_33_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_33_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_32_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_32_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_33_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_33_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¥
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ó	
Î
$__inference_signature_wrapper_141251
conv2d_32_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:
@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_140729o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
)
_user_specified_nameconv2d_32_input

h
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140738

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
e
I__inference_activation_58_layer_call_and_return_conditional_losses_140837

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
í
e
I__inference_activation_57_layer_call_and_return_conditional_losses_140805

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  @:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @
 
_user_specified_nameinputs
Ç
J
.__inference_activation_56_layer_call_fn_141275

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_56_layer_call_and_return_conditional_losses_140781h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿDD@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_33_layer_call_and_return_conditional_losses_141309

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ""@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@
 
_user_specified_nameinputs
µ
G
+__inference_flatten_16_layer_call_fn_141334

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_140814b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
G
Ì
__inference__traced_save_141520
file_prefix/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_32_kernel_m_read_readvariableop4
0savev2_adam_conv2d_32_bias_m_read_readvariableop6
2savev2_adam_conv2d_33_kernel_m_read_readvariableop4
0savev2_adam_conv2d_33_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop6
2savev2_adam_conv2d_32_kernel_v_read_readvariableop4
0savev2_adam_conv2d_32_bias_v_read_readvariableop6
2savev2_adam_conv2d_33_kernel_v_read_readvariableop4
0savev2_adam_conv2d_33_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¯
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_32_kernel_m_read_readvariableop0savev2_adam_conv2d_32_bias_m_read_readvariableop2savev2_adam_conv2d_33_kernel_m_read_readvariableop0savev2_adam_conv2d_33_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop2savev2_adam_conv2d_32_kernel_v_read_readvariableop0savev2_adam_conv2d_32_bias_v_read_readvariableop2savev2_adam_conv2d_33_kernel_v_read_readvariableop0savev2_adam_conv2d_33_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*¡
_input_shapes
: :@:@:@@:@:
@:@:@:: : : : : : : : : :@:@:@@:@:
@:@:@::@:@:@@:@:
@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
@: 

_output_shapes
:@:$  

_output_shapes

:@: !

_output_shapes
::"

_output_shapes
: 
í
e
I__inference_activation_57_layer_call_and_return_conditional_losses_141319

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  @:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @
 
_user_specified_nameinputs
ì

*__inference_conv2d_33_layer_call_fn_141299

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_140794w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ""@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@
 
_user_specified_nameinputs
í
e
I__inference_activation_56_layer_call_and_return_conditional_losses_141280

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿDD@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@
 
_user_specified_nameinputs
È

)__inference_dense_32_layer_call_fn_141349

inputs
unknown:
@
	unknown_0:@
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_140826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_32_layer_call_and_return_conditional_losses_141270

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿFF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_32_layer_call_and_return_conditional_losses_140770

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿFF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs
Ú5

!__inference__wrapped_model_140729
conv2d_32_inputP
6sequential_19_conv2d_32_conv2d_readvariableop_resource:@E
7sequential_19_conv2d_32_biasadd_readvariableop_resource:@P
6sequential_19_conv2d_33_conv2d_readvariableop_resource:@@E
7sequential_19_conv2d_33_biasadd_readvariableop_resource:@I
5sequential_19_dense_32_matmul_readvariableop_resource:
@D
6sequential_19_dense_32_biasadd_readvariableop_resource:@G
5sequential_19_dense_33_matmul_readvariableop_resource:@D
6sequential_19_dense_33_biasadd_readvariableop_resource:
identity¢.sequential_19/conv2d_32/BiasAdd/ReadVariableOp¢-sequential_19/conv2d_32/Conv2D/ReadVariableOp¢.sequential_19/conv2d_33/BiasAdd/ReadVariableOp¢-sequential_19/conv2d_33/Conv2D/ReadVariableOp¢-sequential_19/dense_32/BiasAdd/ReadVariableOp¢,sequential_19/dense_32/MatMul/ReadVariableOp¢-sequential_19/dense_33/BiasAdd/ReadVariableOp¢,sequential_19/dense_33/MatMul/ReadVariableOp¬
-sequential_19/conv2d_32/Conv2D/ReadVariableOpReadVariableOp6sequential_19_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ó
sequential_19/conv2d_32/Conv2DConv2Dconv2d_32_input5sequential_19/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*
paddingVALID*
strides
¢
.sequential_19/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp7sequential_19_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Å
sequential_19/conv2d_32/BiasAddBiasAdd'sequential_19/conv2d_32/Conv2D:output:06sequential_19/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@
 sequential_19/activation_56/ReluRelu(sequential_19/conv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@Î
&sequential_19/max_pooling2d_32/MaxPoolMaxPool.sequential_19/activation_56/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@*
ksize
*
paddingVALID*
strides
¬
-sequential_19/conv2d_33/Conv2D/ReadVariableOpReadVariableOp6sequential_19_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ó
sequential_19/conv2d_33/Conv2DConv2D/sequential_19/max_pooling2d_32/MaxPool:output:05sequential_19/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingVALID*
strides
¢
.sequential_19/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp7sequential_19_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Å
sequential_19/conv2d_33/BiasAddBiasAdd'sequential_19/conv2d_33/Conv2D:output:06sequential_19/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @
 sequential_19/activation_57/ReluRelu(sequential_19/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @Î
&sequential_19/max_pooling2d_33/MaxPoolMaxPool.sequential_19/activation_57/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
o
sequential_19/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  ¹
 sequential_19/flatten_16/ReshapeReshape/sequential_19/max_pooling2d_33/MaxPool:output:0'sequential_19/flatten_16/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,sequential_19/dense_32/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0º
sequential_19/dense_32/MatMulMatMul)sequential_19/flatten_16/Reshape:output:04sequential_19/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
-sequential_19/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0»
sequential_19/dense_32/BiasAddBiasAdd'sequential_19/dense_32/MatMul:product:05sequential_19/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 sequential_19/activation_58/ReluRelu'sequential_19/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
,sequential_19/dense_33/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_33_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¿
sequential_19/dense_33/MatMulMatMul.sequential_19/activation_58/Relu:activations:04sequential_19/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_19/dense_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_19/dense_33/BiasAddBiasAdd'sequential_19/dense_33/MatMul:product:05sequential_19/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#sequential_19/activation_59/SigmoidSigmoid'sequential_19/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_19/activation_59/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp/^sequential_19/conv2d_32/BiasAdd/ReadVariableOp.^sequential_19/conv2d_32/Conv2D/ReadVariableOp/^sequential_19/conv2d_33/BiasAdd/ReadVariableOp.^sequential_19/conv2d_33/Conv2D/ReadVariableOp.^sequential_19/dense_32/BiasAdd/ReadVariableOp-^sequential_19/dense_32/MatMul/ReadVariableOp.^sequential_19/dense_33/BiasAdd/ReadVariableOp-^sequential_19/dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 2`
.sequential_19/conv2d_32/BiasAdd/ReadVariableOp.sequential_19/conv2d_32/BiasAdd/ReadVariableOp2^
-sequential_19/conv2d_32/Conv2D/ReadVariableOp-sequential_19/conv2d_32/Conv2D/ReadVariableOp2`
.sequential_19/conv2d_33/BiasAdd/ReadVariableOp.sequential_19/conv2d_33/BiasAdd/ReadVariableOp2^
-sequential_19/conv2d_33/Conv2D/ReadVariableOp-sequential_19/conv2d_33/Conv2D/ReadVariableOp2^
-sequential_19/dense_32/BiasAdd/ReadVariableOp-sequential_19/dense_32/BiasAdd/ReadVariableOp2\
,sequential_19/dense_32/MatMul/ReadVariableOp,sequential_19/dense_32/MatMul/ReadVariableOp2^
-sequential_19/dense_33/BiasAdd/ReadVariableOp-sequential_19/dense_33/BiasAdd/ReadVariableOp2\
,sequential_19/dense_33/MatMul/ReadVariableOp,sequential_19/dense_33/MatMul/ReadVariableOp:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
)
_user_specified_nameconv2d_32_input
§
J
.__inference_activation_58_layer_call_fn_141364

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_58_layer_call_and_return_conditional_losses_140837`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
£+

I__inference_sequential_19_layer_call_and_return_conditional_losses_141006

inputs*
conv2d_32_140978:@
conv2d_32_140980:@*
conv2d_33_140985:@@
conv2d_33_140987:@#
dense_32_140993:
@
dense_32_140995:@!
dense_33_140999:@
dense_33_141001:
identity¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCallü
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32_140978conv2d_32_140980*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_140770î
activation_56/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿDD@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_56_layer_call_and_return_conditional_losses_140781ð
 max_pooling2d_32/PartitionedCallPartitionedCall&activation_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140738
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0conv2d_33_140985conv2d_33_140987*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_140794î
activation_57/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_57_layer_call_and_return_conditional_losses_140805ð
 max_pooling2d_33/PartitionedCallPartitionedCall&activation_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_140750á
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_140814
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_140993dense_32_140995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_140826å
activation_58/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_58_layer_call_and_return_conditional_losses_140837
 dense_33/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0dense_33_140999dense_33_141001*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_140849å
activation_59/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_59_layer_call_and_return_conditional_losses_140860u
IdentityIdentity&activation_59/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿFF: : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿFF
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*È
serving_default´
S
conv2d_32_input@
!serving_default_conv2d_32_input:0ÿÿÿÿÿÿÿÿÿFFA
activation_590
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:·
Þ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
»

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
ó
_iter

`beta_1

abeta_2
	bdecay
clearning_ratem¬m­)m®*m¯Cm°Dm±Qm²Rm³v´vµ)v¶*v·Cv¸Dv¹QvºRv»"
	optimizer
X
0
1
)2
*3
C4
D5
Q6
R7"
trackable_list_wrapper
X
0
1
)2
*3
C4
D5
Q6
R7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_19_layer_call_fn_140882
.__inference_sequential_19_layer_call_fn_141135
.__inference_sequential_19_layer_call_fn_141156
.__inference_sequential_19_layer_call_fn_141046À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_19_layer_call_and_return_conditional_losses_141192
I__inference_sequential_19_layer_call_and_return_conditional_losses_141228
I__inference_sequential_19_layer_call_and_return_conditional_losses_141077
I__inference_sequential_19_layer_call_and_return_conditional_losses_141108À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÔBÑ
!__inference__wrapped_model_140729conv2d_32_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
iserving_default"
signature_map
*:(@2conv2d_32/kernel
:@2conv2d_32/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv2d_32_layer_call_fn_141260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv2d_32_layer_call_and_return_conditional_losses_141270¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_activation_56_layer_call_fn_141275¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_activation_56_layer_call_and_return_conditional_losses_141280¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_max_pooling2d_32_layer_call_fn_141285¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_141290¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(@@2conv2d_33/kernel
:@2conv2d_33/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv2d_33_layer_call_fn_141299¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv2d_33_layer_call_and_return_conditional_losses_141309¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_activation_57_layer_call_fn_141314¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_activation_57_layer_call_and_return_conditional_losses_141319¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_max_pooling2d_33_layer_call_fn_141324¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_141329¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_16_layer_call_fn_141334¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_16_layer_call_and_return_conditional_losses_141340¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
@2dense_32/kernel
:@2dense_32/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_32_layer_call_fn_141349¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_32_layer_call_and_return_conditional_losses_141359¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_activation_58_layer_call_fn_141364¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_activation_58_layer_call_and_return_conditional_losses_141369¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:@2dense_33/kernel
:2dense_33/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_33_layer_call_fn_141378¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_33_layer_call_and_return_conditional_losses_141388¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_activation_59_layer_call_fn_141393¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_activation_59_layer_call_and_return_conditional_losses_141398¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÓBÐ
$__inference_signature_wrapper_141251conv2d_32_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
R

£total

¤count
¥	variables
¦	keras_api"
_tf_keras_metric
c

§total

¨count
©
_fn_kwargs
ª	variables
«	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
£0
¤1"
trackable_list_wrapper
.
¥	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
§0
¨1"
trackable_list_wrapper
.
ª	variables"
_generic_user_object
/:-@2Adam/conv2d_32/kernel/m
!:@2Adam/conv2d_32/bias/m
/:-@@2Adam/conv2d_33/kernel/m
!:@2Adam/conv2d_33/bias/m
(:&
@2Adam/dense_32/kernel/m
 :@2Adam/dense_32/bias/m
&:$@2Adam/dense_33/kernel/m
 :2Adam/dense_33/bias/m
/:-@2Adam/conv2d_32/kernel/v
!:@2Adam/conv2d_32/bias/v
/:-@@2Adam/conv2d_33/kernel/v
!:@2Adam/conv2d_33/bias/v
(:&
@2Adam/dense_32/kernel/v
 :@2Adam/dense_32/bias/v
&:$@2Adam/dense_33/kernel/v
 :2Adam/dense_33/bias/v±
!__inference__wrapped_model_140729)*CDQR@¢=
6¢3
1.
conv2d_32_inputÿÿÿÿÿÿÿÿÿFF
ª "=ª:
8
activation_59'$
activation_59ÿÿÿÿÿÿÿÿÿµ
I__inference_activation_56_layer_call_and_return_conditional_losses_141280h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿDD@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿDD@
 
.__inference_activation_56_layer_call_fn_141275[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿDD@
ª " ÿÿÿÿÿÿÿÿÿDD@µ
I__inference_activation_57_layer_call_and_return_conditional_losses_141319h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  @
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  @
 
.__inference_activation_57_layer_call_fn_141314[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  @
ª " ÿÿÿÿÿÿÿÿÿ  @¥
I__inference_activation_58_layer_call_and_return_conditional_losses_141369X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
.__inference_activation_58_layer_call_fn_141364K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¥
I__inference_activation_59_layer_call_and_return_conditional_losses_141398X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
.__inference_activation_59_layer_call_fn_141393K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿµ
E__inference_conv2d_32_layer_call_and_return_conditional_losses_141270l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿFF
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿDD@
 
*__inference_conv2d_32_layer_call_fn_141260_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿFF
ª " ÿÿÿÿÿÿÿÿÿDD@µ
E__inference_conv2d_33_layer_call_and_return_conditional_losses_141309l)*7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ""@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  @
 
*__inference_conv2d_33_layer_call_fn_141299_)*7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ""@
ª " ÿÿÿÿÿÿÿÿÿ  @¦
D__inference_dense_32_layer_call_and_return_conditional_losses_141359^CD1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
)__inference_dense_32_layer_call_fn_141349QCD1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_33_layer_call_and_return_conditional_losses_141388\QR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_33_layer_call_fn_141378OQR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¬
F__inference_flatten_16_layer_call_and_return_conditional_losses_141340b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_flatten_16_layer_call_fn_141334U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿï
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_141290R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_32_layer_call_fn_141285R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_141329R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_33_layer_call_fn_141324R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
I__inference_sequential_19_layer_call_and_return_conditional_losses_141077{)*CDQRH¢E
>¢;
1.
conv2d_32_inputÿÿÿÿÿÿÿÿÿFF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
I__inference_sequential_19_layer_call_and_return_conditional_losses_141108{)*CDQRH¢E
>¢;
1.
conv2d_32_inputÿÿÿÿÿÿÿÿÿFF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
I__inference_sequential_19_layer_call_and_return_conditional_losses_141192r)*CDQR?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿFF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
I__inference_sequential_19_layer_call_and_return_conditional_losses_141228r)*CDQR?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿFF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
.__inference_sequential_19_layer_call_fn_140882n)*CDQRH¢E
>¢;
1.
conv2d_32_inputÿÿÿÿÿÿÿÿÿFF
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
.__inference_sequential_19_layer_call_fn_141046n)*CDQRH¢E
>¢;
1.
conv2d_32_inputÿÿÿÿÿÿÿÿÿFF
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_19_layer_call_fn_141135e)*CDQR?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿFF
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_19_layer_call_fn_141156e)*CDQR?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿFF
p

 
ª "ÿÿÿÿÿÿÿÿÿÇ
$__inference_signature_wrapper_141251)*CDQRS¢P
¢ 
IªF
D
conv2d_32_input1.
conv2d_32_inputÿÿÿÿÿÿÿÿÿFF"=ª:
8
activation_59'$
activation_59ÿÿÿÿÿÿÿÿÿ