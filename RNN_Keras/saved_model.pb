Ý<
Õ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
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
P
Shape

input"T
output"out_type"	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
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
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ä9

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
:*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
:*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:*
dtype0

batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_14/gamma

0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes	
:*
dtype0

batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_14/beta

/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes	
:*
dtype0

"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_14/moving_mean

6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_14/moving_variance

:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes	
:*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	 *
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
: *
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

: *
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
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

lstm_12/lstm_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_12/lstm_cell_12/kernel

/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/kernel*
_output_shapes
:	*
dtype0
¨
%lstm_12/lstm_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_12/lstm_cell_12/recurrent_kernel
¡
9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_12/lstm_cell_12/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_12/lstm_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_12/lstm_cell_12/bias

-lstm_12/lstm_cell_12/bias/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/bias*
_output_shapes	
:*
dtype0

lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namelstm_13/lstm_cell_13/kernel

/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/kernel* 
_output_shapes
:
*
dtype0
¨
%lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_13/lstm_cell_13/recurrent_kernel
¡
9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_13/lstm_cell_13/bias

-lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/bias*
_output_shapes	
:*
dtype0

lstm_14/lstm_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namelstm_14/lstm_cell_14/kernel

/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/kernel* 
_output_shapes
:
*
dtype0
¨
%lstm_14/lstm_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_14/lstm_cell_14/recurrent_kernel
¡
9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_14/lstm_cell_14/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_14/lstm_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_14/lstm_cell_14/bias

-lstm_14/lstm_cell_14/bias/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/bias*
_output_shapes	
:*
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

#Adam/batch_normalization_12/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/m

7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_12/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/m

6Adam/batch_normalization_12/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_13/gamma/m

7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_13/beta/m

6Adam/batch_normalization_13/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_14/gamma/m

7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_14/beta/m

6Adam/batch_normalization_14/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	 *
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
: *
dtype0

Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_12/lstm_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/m

6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/m*
_output_shapes
:	*
dtype0
¶
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
¯
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

 Adam/lstm_12/lstm_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/m

4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/m*
_output_shapes	
:*
dtype0
¢
"Adam/lstm_13/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/m

6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/m* 
_output_shapes
:
*
dtype0
¶
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
¯
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

 Adam/lstm_13/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/m

4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/m*
_output_shapes	
:*
dtype0
¢
"Adam/lstm_14/lstm_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/lstm_14/lstm_cell_14/kernel/m

6Adam/lstm_14/lstm_cell_14/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_14/kernel/m* 
_output_shapes
:
*
dtype0
¶
,Adam/lstm_14/lstm_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m
¯
@Adam/lstm_14/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

 Adam/lstm_14/lstm_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_14/lstm_cell_14/bias/m

4Adam/lstm_14/lstm_cell_14/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_14/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_12/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/v

7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_12/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/v

6Adam/batch_normalization_12/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_13/gamma/v

7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_13/beta/v

6Adam/batch_normalization_13/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_14/gamma/v

7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_14/beta/v

6Adam/batch_normalization_14/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	 *
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
: *
dtype0

Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_12/lstm_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/v

6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/v*
_output_shapes
:	*
dtype0
¶
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
¯
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

 Adam/lstm_12/lstm_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/v

4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/v*
_output_shapes	
:*
dtype0
¢
"Adam/lstm_13/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/v

6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/v* 
_output_shapes
:
*
dtype0
¶
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
¯
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

 Adam/lstm_13/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/v

4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/v*
_output_shapes	
:*
dtype0
¢
"Adam/lstm_14/lstm_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/lstm_14/lstm_cell_14/kernel/v

6Adam/lstm_14/lstm_cell_14/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_14/kernel/v* 
_output_shapes
:
*
dtype0
¶
,Adam/lstm_14/lstm_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v
¯
@Adam/lstm_14/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

 Adam/lstm_14/lstm_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_14/lstm_cell_14/bias/v

4Adam/lstm_14/lstm_cell_14/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_14/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ï
valueÄBÀ B¸
º
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
¥
	variables
 trainable_variables
!regularization_losses
"	keras_api
#_random_generator
$__call__
*%&call_and_return_all_conditional_losses* 
Õ
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
Á
1cell
2
state_spec
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses*
¥
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses* 
Õ
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
Á
Lcell
M
state_spec
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R_random_generator
S__call__
*T&call_and_return_all_conditional_losses*
¥
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses* 
Õ
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
¦

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
¥
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses* 
¦

vkernel
wbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses*
Õ
~iter

beta_1
beta_2

decay
learning_rate'm(mBmCm]m^mgmhmvmwm	m	m	m	m	m	m	m	m	m'v(vBvCv]v^vgvhvvvwv	v 	v¡	v¢	v£	v¤	v¥	v¦	v§	v¨*
Ë
0
1
2
'3
(4
)5
*6
7
8
9
B10
C11
D12
E13
14
15
16
]17
^18
_19
`20
g21
h22
v23
w24*

0
1
2
'3
(4
5
6
7
B8
C9
10
11
12
]13
^14
g15
h16
v17
w18*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
î

state_size
kernel
recurrent_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses*
* 

0
1
2*

0
1
2*
* 
¥
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
 trainable_variables
!regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 
* 
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
'0
(1
)2
*3*

'0
(1*
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
î
ª
state_size
kernel
recurrent_kernel
	bias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯_random_generator
°__call__
+±&call_and_return_all_conditional_losses*
* 

0
1
2*

0
1
2*
* 
¥
²states
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 
* 
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
B0
C1
D2
E3*

B0
C1*
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
î
Â
state_size
kernel
recurrent_kernel
	bias
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç_random_generator
È__call__
+É&call_and_return_all_conditional_losses*
* 

0
1
2*

0
1
2*
* 
¥
Êstates
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
]0
^1
_2
`3*

]0
^1*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1*

g0
h1*
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

v0
w1*

v0
w1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*
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
[U
VARIABLE_VALUElstm_12/lstm_cell_12/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_12/lstm_cell_12/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_12/lstm_cell_12/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_13/lstm_cell_13/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_13/lstm_cell_13/recurrent_kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_13/lstm_cell_13/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUElstm_14/lstm_cell_14/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%lstm_14/lstm_cell_14/recurrent_kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElstm_14/lstm_cell_14/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
.
)0
*1
D2
E3
_4
`5*
Z
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
10
11*

é0
ê1*
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 

)0
*1*
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

10*
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1*
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

L0*
* 
* 
* 
* 
* 
* 
* 
* 

_0
`1*
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

útotal

ûcount
ü	variables
ý	keras_api*
M

þtotal

ÿcount

_fn_kwargs
	variables
	keras_api*
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
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ú0
û1*

ü	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

þ0
ÿ1*

	variables*

VARIABLE_VALUE#Adam/batch_normalization_12/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_12/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_13/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_13/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_14/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_14/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_14/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_14/lstm_cell_14/recurrent_kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/lstm_14/lstm_cell_14/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_12/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_12/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_13/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_13/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_14/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_14/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_14/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_14/lstm_cell_14/recurrent_kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/lstm_14/lstm_cell_14/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_lstm_12_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ<
Ø
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_12_inputlstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/bias&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betalstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/bias&batch_normalization_13/moving_variancebatch_normalization_13/gamma"batch_normalization_13/moving_meanbatch_normalization_13/betalstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/bias&batch_normalization_14/moving_variancebatch_normalization_14/gamma"batch_normalization_14/moving_meanbatch_normalization_14/betadense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_166651
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ô
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOp9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOp-lstm_12/lstm_cell_12/bias/Read/ReadVariableOp/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOp9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp-lstm_13/lstm_cell_13/bias/Read/ReadVariableOp/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOp9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOp-lstm_14/lstm_cell_14/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_12/beta/m/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_13/beta/m/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_14/beta/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_14/kernel/m/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_14/bias/m/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_12/beta/v/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_13/beta/v/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_14/beta/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_14/kernel/v/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_14/bias/v/Read/ReadVariableOpConst*U
TinN
L2J	*
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
__inference__traced_save_169420
Ï
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancebatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variancebatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biaslstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biastotalcounttotal_1count_1#Adam/batch_normalization_12/gamma/m"Adam/batch_normalization_12/beta/m#Adam/batch_normalization_13/gamma/m"Adam/batch_normalization_13/beta/m#Adam/batch_normalization_14/gamma/m"Adam/batch_normalization_14/beta/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m"Adam/lstm_12/lstm_cell_12/kernel/m,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m Adam/lstm_12/lstm_cell_12/bias/m"Adam/lstm_13/lstm_cell_13/kernel/m,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m Adam/lstm_13/lstm_cell_13/bias/m"Adam/lstm_14/lstm_cell_14/kernel/m,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m Adam/lstm_14/lstm_cell_14/bias/m#Adam/batch_normalization_12/gamma/v"Adam/batch_normalization_12/beta/v#Adam/batch_normalization_13/gamma/v"Adam/batch_normalization_13/beta/v#Adam/batch_normalization_14/gamma/v"Adam/batch_normalization_14/beta/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v"Adam/lstm_12/lstm_cell_12/kernel/v,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v Adam/lstm_12/lstm_cell_12/bias/v"Adam/lstm_13/lstm_cell_13/kernel/v,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v Adam/lstm_13/lstm_cell_13/bias/v"Adam/lstm_14/lstm_cell_14/kernel/v,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v Adam/lstm_14/lstm_cell_14/bias/v*T
TinM
K2I*
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
"__inference__traced_restore_1696466
í
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_164026

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
µ
G
+__inference_dropout_17_layer_call_fn_167995

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_164192e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
§8

C__inference_lstm_13_layer_call_and_return_conditional_losses_163150

inputs'
lstm_cell_13_163068:
'
lstm_cell_13_163070:
"
lstm_cell_13_163072:	
identity¢$lstm_cell_13/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskø
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_163068lstm_cell_13_163070lstm_cell_13_163072*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163067n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_163068lstm_cell_13_163070lstm_cell_13_163072*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163081*
condR
while_cond_163080*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
îA
Ò

lstm_12_while_body_165609,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	Q
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
K
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	O
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
I
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	¢1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp¢0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp¢2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: q
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: ±
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_12/while/NoOpNoOp2^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"È
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2d
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2h
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
Ö
7__inference_batch_normalization_14_layer_call_fn_168766

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163853p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

·
(__inference_lstm_13_layer_call_fn_167407

inputs
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_164179t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Í=

H__inference_sequential_4_layer_call_and_return_conditional_losses_164411

inputs!
lstm_12_164014:	"
lstm_12_164016:

lstm_12_164018:	,
batch_normalization_12_164028:	,
batch_normalization_12_164030:	,
batch_normalization_12_164032:	,
batch_normalization_12_164034:	"
lstm_13_164180:
"
lstm_13_164182:

lstm_13_164184:	,
batch_normalization_13_164194:	,
batch_normalization_13_164196:	,
batch_normalization_13_164198:	,
batch_normalization_13_164200:	"
lstm_14_164346:
"
lstm_14_164348:

lstm_14_164350:	,
batch_normalization_14_164360:	,
batch_normalization_14_164362:	,
batch_normalization_14_164364:	,
batch_normalization_14_164366:	!
dense_8_164381:	 
dense_8_164383:  
dense_9_164405: 
dense_9_164407:
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_164014lstm_12_164016lstm_12_164018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_164013ã
dropout_16/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_164026
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0batch_normalization_12_164028batch_normalization_12_164030batch_normalization_12_164032batch_normalization_12_164034*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162942´
lstm_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0lstm_13_164180lstm_13_164182lstm_13_164184*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_164179ã
dropout_17/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_164192
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0batch_normalization_13_164194batch_normalization_13_164196batch_normalization_13_164198batch_normalization_13_164200*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163374°
lstm_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0lstm_14_164346lstm_14_164348lstm_14_164350*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_164345ß
dropout_18/PartitionedCallPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_164358
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0batch_normalization_14_164360batch_normalization_14_164362batch_normalization_14_164364batch_normalization_14_164366*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163806
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_8_164381dense_8_164383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_164380Þ
dropout_19/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_164391
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_164405dense_9_164407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_164404w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
K

C__inference_lstm_13_layer_call_and_return_conditional_losses_167704
inputs_0?
+lstm_cell_13_matmul_readvariableop_resource:
A
-lstm_cell_13_matmul_1_readvariableop_resource:
;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_167620*
condR
while_cond_167619*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

·
(__inference_lstm_13_layer_call_fn_167418

inputs
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_164874t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs

¶
(__inference_lstm_12_layer_call_fn_166684

inputs
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_164013t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
í
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_164192

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¯
¹
(__inference_lstm_13_layer_call_fn_167396
inputs_0
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_163341}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¡
G
+__inference_dropout_19_layer_call_fn_168845

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_164391`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
À

(__inference_dense_9_layer_call_fn_168876

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_164404o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
îA
Ò

lstm_12_while_body_166096,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	Q
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
K
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	O
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
I
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	¢1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp¢0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp¢2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ò
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: q
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: ±
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_12/while/NoOpNoOp2^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"È
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2d
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2h
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ï

H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163499

inputs

states
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
ÄJ

C__inference_lstm_13_layer_call_and_return_conditional_losses_164874

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
A
-lstm_cell_13_matmul_1_readvariableop_resource:
;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164790*
condR
while_cond_164789*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
µ8
Ò
while_body_167040
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
A
2while_lstm_cell_12_biasadd_readvariableop_resource:	¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
á
µ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_168786

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æC

H__inference_sequential_4_layer_call_and_return_conditional_losses_165194

inputs!
lstm_12_165131:	"
lstm_12_165133:

lstm_12_165135:	,
batch_normalization_12_165139:	,
batch_normalization_12_165141:	,
batch_normalization_12_165143:	,
batch_normalization_12_165145:	"
lstm_13_165148:
"
lstm_13_165150:

lstm_13_165152:	,
batch_normalization_13_165156:	,
batch_normalization_13_165158:	,
batch_normalization_13_165160:	,
batch_normalization_13_165162:	"
lstm_14_165165:
"
lstm_14_165167:

lstm_14_165169:	,
batch_normalization_14_165173:	,
batch_normalization_14_165175:	,
batch_normalization_14_165177:	,
batch_normalization_14_165179:	!
dense_8_165182:	 
dense_8_165184:  
dense_9_165188: 
dense_9_165190:
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢"dropout_18/StatefulPartitionedCall¢"dropout_19/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_165131lstm_12_165133lstm_12_165135*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_165062ó
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_164903
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0batch_normalization_12_165139batch_normalization_12_165141batch_normalization_12_165143batch_normalization_12_165145*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162989´
lstm_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0lstm_13_165148lstm_13_165150lstm_13_165152*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_164874
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_164715
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0batch_normalization_13_165156batch_normalization_13_165158batch_normalization_13_165160batch_normalization_13_165162*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163421°
lstm_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0lstm_14_165165lstm_14_165167lstm_14_165169*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_164686
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_164527
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0batch_normalization_14_165173batch_normalization_14_165175batch_normalization_14_165177batch_normalization_14_165179*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163853
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_8_165182dense_8_165184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_164380
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_164494
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_165188dense_9_165190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_164404w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹
Ã
while_cond_163271
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163271___redundant_placeholder04
0while_while_cond_163271___redundant_placeholder14
0while_while_cond_163271___redundant_placeholder24
0while_while_cond_163271___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
µ8
Ò
while_body_164978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
A
2while_lstm_cell_12_biasadd_readvariableop_resource:	¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Q
õ
&sequential_4_lstm_14_while_body_162452F
Bsequential_4_lstm_14_while_sequential_4_lstm_14_while_loop_counterL
Hsequential_4_lstm_14_while_sequential_4_lstm_14_while_maximum_iterations*
&sequential_4_lstm_14_while_placeholder,
(sequential_4_lstm_14_while_placeholder_1,
(sequential_4_lstm_14_while_placeholder_2,
(sequential_4_lstm_14_while_placeholder_3E
Asequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1_0
}sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:
^
Jsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
X
Isequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	'
#sequential_4_lstm_14_while_identity)
%sequential_4_lstm_14_while_identity_1)
%sequential_4_lstm_14_while_identity_2)
%sequential_4_lstm_14_while_identity_3)
%sequential_4_lstm_14_while_identity_4)
%sequential_4_lstm_14_while_identity_5C
?sequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1
{sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:
\
Hsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
V
Gsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	¢>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp¢=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp¢?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp
Lsequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
>sequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_14_while_placeholderUsequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0È
=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpHsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ù
.sequential_4/lstm_14/while/lstm_cell_14/MatMulMatMulEsequential_4/lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpJsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0à
0sequential_4/lstm_14/while/lstm_cell_14/MatMul_1MatMul(sequential_4_lstm_14_while_placeholder_2Gsequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+sequential_4/lstm_14/while/lstm_cell_14/addAddV28sequential_4/lstm_14/while/lstm_cell_14/MatMul:product:0:sequential_4/lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpIsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0æ
/sequential_4/lstm_14/while/lstm_cell_14/BiasAddBiasAdd/sequential_4/lstm_14/while/lstm_cell_14/add:z:0Fsequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7sequential_4/lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :²
-sequential_4/lstm_14/while/lstm_cell_14/splitSplit@sequential_4/lstm_14/while/lstm_cell_14/split/split_dim:output:08sequential_4/lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¥
/sequential_4/lstm_14/while/lstm_cell_14/SigmoidSigmoid6sequential_4/lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid6sequential_4/lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
+sequential_4/lstm_14/while/lstm_cell_14/mulMul5sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_1:y:0(sequential_4_lstm_14_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_4/lstm_14/while/lstm_cell_14/ReluRelu6sequential_4/lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
-sequential_4/lstm_14/while/lstm_cell_14/mul_1Mul3sequential_4/lstm_14/while/lstm_cell_14/Sigmoid:y:0:sequential_4/lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
-sequential_4/lstm_14/while/lstm_cell_14/add_1AddV2/sequential_4/lstm_14/while/lstm_cell_14/mul:z:01sequential_4/lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid6sequential_4/lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.sequential_4/lstm_14/while/lstm_cell_14/Relu_1Relu1sequential_4/lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
-sequential_4/lstm_14/while/lstm_cell_14/mul_2Mul5sequential_4/lstm_14/while/lstm_cell_14/Sigmoid_2:y:0<sequential_4/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?sequential_4/lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_14_while_placeholder_1&sequential_4_lstm_14_while_placeholder1sequential_4/lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 sequential_4/lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_4/lstm_14/while/addAddV2&sequential_4_lstm_14_while_placeholder)sequential_4/lstm_14/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_4/lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 sequential_4/lstm_14/while/add_1AddV2Bsequential_4_lstm_14_while_sequential_4_lstm_14_while_loop_counter+sequential_4/lstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 
#sequential_4/lstm_14/while/IdentityIdentity$sequential_4/lstm_14/while/add_1:z:0 ^sequential_4/lstm_14/while/NoOp*
T0*
_output_shapes
: ¾
%sequential_4/lstm_14/while/Identity_1IdentityHsequential_4_lstm_14_while_sequential_4_lstm_14_while_maximum_iterations ^sequential_4/lstm_14/while/NoOp*
T0*
_output_shapes
: 
%sequential_4/lstm_14/while/Identity_2Identity"sequential_4/lstm_14/while/add:z:0 ^sequential_4/lstm_14/while/NoOp*
T0*
_output_shapes
: Ø
%sequential_4/lstm_14/while/Identity_3IdentityOsequential_4/lstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_4/lstm_14/while/NoOp*
T0*
_output_shapes
: :éèÒ¹
%sequential_4/lstm_14/while/Identity_4Identity1sequential_4/lstm_14/while/lstm_cell_14/mul_2:z:0 ^sequential_4/lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%sequential_4/lstm_14/while/Identity_5Identity1sequential_4/lstm_14/while/lstm_cell_14/add_1:z:0 ^sequential_4/lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
sequential_4/lstm_14/while/NoOpNoOp?^sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_4_lstm_14_while_identity,sequential_4/lstm_14/while/Identity:output:0"W
%sequential_4_lstm_14_while_identity_1.sequential_4/lstm_14/while/Identity_1:output:0"W
%sequential_4_lstm_14_while_identity_2.sequential_4/lstm_14/while/Identity_2:output:0"W
%sequential_4_lstm_14_while_identity_3.sequential_4/lstm_14/while/Identity_3:output:0"W
%sequential_4_lstm_14_while_identity_4.sequential_4/lstm_14/while/Identity_4:output:0"W
%sequential_4_lstm_14_while_identity_5.sequential_4/lstm_14/while/Identity_5:output:0"
Gsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resourceIsequential_4_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"
Hsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resourceJsequential_4_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"
Fsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resourceHsequential_4_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"
?sequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1Asequential_4_lstm_14_while_sequential_4_lstm_14_strided_slice_1_0"ü
{sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>sequential_4/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2~
=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp=sequential_4/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2
?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?sequential_4/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
í
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_168005

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
ÉJ

C__inference_lstm_14_layer_call_and_return_conditional_losses_164345

inputs?
+lstm_cell_14_matmul_readvariableop_resource:
A
-lstm_cell_14_matmul_1_readvariableop_resource:
;
,lstm_cell_14_biasadd_readvariableop_resource:	
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164261*
condR
while_cond_164260*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¢8

C__inference_lstm_12_layer_call_and_return_conditional_losses_162909

inputs&
lstm_cell_12_162827:	'
lstm_cell_12_162829:
"
lstm_cell_12_162831:	
identity¢$lstm_cell_12/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskø
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_162827lstm_cell_12_162829lstm_cell_12_162831*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162781n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_162827lstm_cell_12_162829lstm_cell_12_162831*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_162840*
condR
while_cond_162839*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

(__inference_dense_8_layer_call_fn_168829

inputs
unknown:	 
	unknown_0: 
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_164380o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ã
while_cond_163512
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163512___redundant_placeholder04
0while_while_cond_163512___redundant_placeholder14
0while_while_cond_163512___redundant_placeholder24
0while_while_cond_163512___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
µ8
Ò
while_body_163929
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
A
2while_lstm_cell_12_biasadd_readvariableop_resource:	¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹8
Ô
while_body_168200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_14_matmul_readvariableop_resource_0:
I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_14_matmul_readvariableop_resource:
G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
A
2while_lstm_cell_14_biasadd_readvariableop_resource:	¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹
Ã
while_cond_167476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167476___redundant_placeholder04
0while_while_cond_167476___redundant_placeholder14
0while_while_cond_167476___redundant_placeholder24
0while_while_cond_167476___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹8
Ô
while_body_164790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
A
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
£
µ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163374

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_164358

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹8
Ô
while_body_164261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_14_matmul_readvariableop_resource_0:
I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_14_matmul_readvariableop_resource:
G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
A
2while_lstm_cell_14_biasadd_readvariableop_resource:	¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¯
¹
(__inference_lstm_13_layer_call_fn_167385
inputs_0
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_163150}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
µ8
Ò
while_body_167183
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
A
2while_lstm_cell_12_biasadd_readvariableop_resource:	¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ï

H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163645

inputs

states
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Õ

H__inference_sequential_4_layer_call_and_return_conditional_losses_166037

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	I
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
C
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	G
8batch_normalization_12_batchnorm_readvariableop_resource:	K
<batch_normalization_12_batchnorm_mul_readvariableop_resource:	I
:batch_normalization_12_batchnorm_readvariableop_1_resource:	I
:batch_normalization_12_batchnorm_readvariableop_2_resource:	G
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:
I
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
C
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	G
8batch_normalization_13_batchnorm_readvariableop_resource:	K
<batch_normalization_13_batchnorm_mul_readvariableop_resource:	I
:batch_normalization_13_batchnorm_readvariableop_1_resource:	I
:batch_normalization_13_batchnorm_readvariableop_2_resource:	G
3lstm_14_lstm_cell_14_matmul_readvariableop_resource:
I
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
C
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	G
8batch_normalization_14_batchnorm_readvariableop_resource:	K
<batch_normalization_14_batchnorm_mul_readvariableop_resource:	I
:batch_normalization_14_batchnorm_readvariableop_1_resource:	I
:batch_normalization_14_batchnorm_readvariableop_2_resource:	9
&dense_8_matmul_readvariableop_resource:	 5
'dense_8_biasadd_readvariableop_resource: 8
&dense_9_matmul_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource:
identity¢/batch_normalization_12/batchnorm/ReadVariableOp¢1batch_normalization_12/batchnorm/ReadVariableOp_1¢1batch_normalization_12/batchnorm/ReadVariableOp_2¢3batch_normalization_12/batchnorm/mul/ReadVariableOp¢/batch_normalization_13/batchnorm/ReadVariableOp¢1batch_normalization_13/batchnorm/ReadVariableOp_1¢1batch_normalization_13/batchnorm/ReadVariableOp_2¢3batch_normalization_13/batchnorm/mul/ReadVariableOp¢/batch_normalization_14/batchnorm/ReadVariableOp¢1batch_normalization_14/batchnorm/ReadVariableOp_1¢1batch_normalization_14/batchnorm/ReadVariableOp_2¢3batch_normalization_14/batchnorm/mul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp¢*lstm_12/lstm_cell_12/MatMul/ReadVariableOp¢,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp¢lstm_12/while¢+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢*lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢lstm_13/while¢+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp¢*lstm_14/lstm_cell_14/MatMul/ReadVariableOp¢,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp¢lstm_14/whileC
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:<ÿÿÿÿÿÿÿÿÿT
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:g
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_12_while_body_165609*%
condR
lstm_12_while_cond_165608*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Û
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<c
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
dropout_16/IdentityIdentitylstm_12/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<¥
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0k
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:½
$batch_normalization_12/batchnorm/addAddV27batch_normalization_12/batchnorm/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¬
&batch_normalization_12/batchnorm/mul_1Muldropout_16/Identity:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<©
1batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0¸
&batch_normalization_12/batchnorm/mul_2Mul9batch_normalization_12/batchnorm/ReadVariableOp_1:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
:©
1batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0¸
$batch_normalization_12/batchnorm/subSub9batch_normalization_12/batchnorm/ReadVariableOp_2:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¼
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<g
lstm_13/ShapeShape*batch_normalization_12/batchnorm/add_1:z:0*
T0*
_output_shapes
:e
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¢
lstm_13/transpose	Transpose*batch_normalization_12/batchnorm/add_1:z:0lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿT
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:g
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask 
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0®
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_13_while_body_165765*%
condR
lstm_13_while_cond_165764*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Û
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<c
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
dropout_17/IdentityIdentitylstm_13/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<¥
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0k
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:½
$batch_normalization_13/batchnorm/addAddV27batch_normalization_13/batchnorm/ReadVariableOp:value:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¬
&batch_normalization_13/batchnorm/mul_1Muldropout_17/Identity:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<©
1batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0¸
&batch_normalization_13/batchnorm/mul_2Mul9batch_normalization_13/batchnorm/ReadVariableOp_1:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes	
:©
1batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0¸
$batch_normalization_13/batchnorm/subSub9batch_normalization_13/batchnorm/ReadVariableOp_2:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¼
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<g
lstm_14/ShapeShape*batch_normalization_13/batchnorm/add_1:z:0*
T0*
_output_shapes
:e
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¢
lstm_14/transpose	Transpose*batch_normalization_13/batchnorm/add_1:z:0lstm_14/transpose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿT
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:g
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask 
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0®
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_14_while_body_165921*%
condR
lstm_14_while_cond_165920*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Û
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<c
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
dropout_18/IdentityIdentity lstm_14/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0k
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:½
$batch_normalization_14/batchnorm/addAddV27batch_normalization_14/batchnorm/ReadVariableOp:value:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:0;batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¨
&batch_normalization_14/batchnorm/mul_1Muldropout_18/Identity:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
1batch_normalization_14/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_14_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0¸
&batch_normalization_14/batchnorm/mul_2Mul9batch_normalization_14/batchnorm/ReadVariableOp_1:value:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes	
:©
1batch_normalization_14/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_14_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0¸
$batch_normalization_14/batchnorm/subSub9batch_normalization_14/batchnorm/ReadVariableOp_2:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense_8/MatMulMatMul*batch_normalization_14/batchnorm/add_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
dropout_19/IdentityIdentitydense_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_9/MatMulMatMuldropout_19/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp0^batch_normalization_12/batchnorm/ReadVariableOp2^batch_normalization_12/batchnorm/ReadVariableOp_12^batch_normalization_12/batchnorm/ReadVariableOp_24^batch_normalization_12/batchnorm/mul/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp2^batch_normalization_13/batchnorm/ReadVariableOp_12^batch_normalization_13/batchnorm/ReadVariableOp_24^batch_normalization_13/batchnorm/mul/ReadVariableOp0^batch_normalization_14/batchnorm/ReadVariableOp2^batch_normalization_14/batchnorm/ReadVariableOp_12^batch_normalization_14/batchnorm/ReadVariableOp_24^batch_normalization_14/batchnorm/mul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2f
1batch_normalization_12/batchnorm/ReadVariableOp_11batch_normalization_12/batchnorm/ReadVariableOp_12f
1batch_normalization_12/batchnorm/ReadVariableOp_21batch_normalization_12/batchnorm/ReadVariableOp_22j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2b
/batch_normalization_13/batchnorm/ReadVariableOp/batch_normalization_13/batchnorm/ReadVariableOp2f
1batch_normalization_13/batchnorm/ReadVariableOp_11batch_normalization_13/batchnorm/ReadVariableOp_12f
1batch_normalization_13/batchnorm/ReadVariableOp_21batch_normalization_13/batchnorm/ReadVariableOp_22j
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2b
/batch_normalization_14/batchnorm/ReadVariableOp/batch_normalization_14/batchnorm/ReadVariableOp2f
1batch_normalization_14/batchnorm/ReadVariableOp_11batch_normalization_14/batchnorm/ReadVariableOp_12f
1batch_normalization_14/batchnorm/ReadVariableOp_21batch_normalization_14/batchnorm/ReadVariableOp_22j
3batch_normalization_14/batchnorm/mul/ReadVariableOp3batch_normalization_14/batchnorm/mul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2Z
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2X
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp*lstm_12/lstm_cell_12/MatMul/ReadVariableOp2\
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp2
lstm_12/whilelstm_12/while2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while2Z
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp*lstm_14/lstm_cell_14/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
µ8
Ò
while_body_166897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
A
2while_lstm_cell_12_biasadd_readvariableop_resource:	¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
òA
Ô

lstm_14_while_body_165921,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:
Q
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
K
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorM
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:
O
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
I
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	¢1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp¢0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp¢2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0®
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ò
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: q
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: ±
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_14/while/NoOpNoOp2^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"È
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ù
ø
-__inference_lstm_cell_14_layer_call_fn_169100

inputs
states_0
states_1
unknown:

	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163499p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
½

-__inference_sequential_4_layer_call_fn_165495

inputs
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:

	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:


unknown_14:


unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	 

unknown_21: 

unknown_22: 

unknown_23:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_164411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
µ
G
+__inference_dropout_16_layer_call_fn_167272

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_164026e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
â=

H__inference_sequential_4_layer_call_and_return_conditional_losses_165368
lstm_12_input!
lstm_12_165305:	"
lstm_12_165307:

lstm_12_165309:	,
batch_normalization_12_165313:	,
batch_normalization_12_165315:	,
batch_normalization_12_165317:	,
batch_normalization_12_165319:	"
lstm_13_165322:
"
lstm_13_165324:

lstm_13_165326:	,
batch_normalization_13_165330:	,
batch_normalization_13_165332:	,
batch_normalization_13_165334:	,
batch_normalization_13_165336:	"
lstm_14_165339:
"
lstm_14_165341:

lstm_14_165343:	,
batch_normalization_14_165347:	,
batch_normalization_14_165349:	,
batch_normalization_14_165351:	,
batch_normalization_14_165353:	!
dense_8_165356:	 
dense_8_165358:  
dense_9_165362: 
dense_9_165364:
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_165305lstm_12_165307lstm_12_165309*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_164013ã
dropout_16/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_164026
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0batch_normalization_12_165313batch_normalization_12_165315batch_normalization_12_165317batch_normalization_12_165319*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162942´
lstm_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0lstm_13_165322lstm_13_165324lstm_13_165326*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_164179ã
dropout_17/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_164192
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0batch_normalization_13_165330batch_normalization_13_165332batch_normalization_13_165334batch_normalization_13_165336*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163374°
lstm_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0lstm_14_165339lstm_14_165341lstm_14_165343*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_164345ß
dropout_18/PartitionedCallPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_164358
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0batch_normalization_14_165347batch_normalization_14_165349batch_normalization_14_165351batch_normalization_14_165353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163806
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_8_165356dense_8_165358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_164380Þ
dropout_19/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_164391
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_165362dense_9_165364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_164404w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
'
_user_specified_namelstm_12_input
ÖÂ
Ð
H__inference_sequential_4_layer_call_and_return_conditional_losses_166594

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	I
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
C
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	M
>batch_normalization_12_assignmovingavg_readvariableop_resource:	O
@batch_normalization_12_assignmovingavg_1_readvariableop_resource:	K
<batch_normalization_12_batchnorm_mul_readvariableop_resource:	G
8batch_normalization_12_batchnorm_readvariableop_resource:	G
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:
I
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
C
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	M
>batch_normalization_13_assignmovingavg_readvariableop_resource:	O
@batch_normalization_13_assignmovingavg_1_readvariableop_resource:	K
<batch_normalization_13_batchnorm_mul_readvariableop_resource:	G
8batch_normalization_13_batchnorm_readvariableop_resource:	G
3lstm_14_lstm_cell_14_matmul_readvariableop_resource:
I
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
C
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	M
>batch_normalization_14_assignmovingavg_readvariableop_resource:	O
@batch_normalization_14_assignmovingavg_1_readvariableop_resource:	K
<batch_normalization_14_batchnorm_mul_readvariableop_resource:	G
8batch_normalization_14_batchnorm_readvariableop_resource:	9
&dense_8_matmul_readvariableop_resource:	 5
'dense_8_biasadd_readvariableop_resource: 8
&dense_9_matmul_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource:
identity¢&batch_normalization_12/AssignMovingAvg¢5batch_normalization_12/AssignMovingAvg/ReadVariableOp¢(batch_normalization_12/AssignMovingAvg_1¢7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_12/batchnorm/ReadVariableOp¢3batch_normalization_12/batchnorm/mul/ReadVariableOp¢&batch_normalization_13/AssignMovingAvg¢5batch_normalization_13/AssignMovingAvg/ReadVariableOp¢(batch_normalization_13/AssignMovingAvg_1¢7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_13/batchnorm/ReadVariableOp¢3batch_normalization_13/batchnorm/mul/ReadVariableOp¢&batch_normalization_14/AssignMovingAvg¢5batch_normalization_14/AssignMovingAvg/ReadVariableOp¢(batch_normalization_14/AssignMovingAvg_1¢7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_14/batchnorm/ReadVariableOp¢3batch_normalization_14/batchnorm/mul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp¢*lstm_12/lstm_cell_12/MatMul/ReadVariableOp¢,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp¢lstm_12/while¢+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢*lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢lstm_13/while¢+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp¢*lstm_14/lstm_cell_14/MatMul/ReadVariableOp¢,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp¢lstm_14/whileC
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:<ÿÿÿÿÿÿÿÿÿT
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:g
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_12_while_body_166096*%
condR
lstm_12_while_cond_166095*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Û
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<c
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_16/dropout/MulMullstm_12/transpose_1:y:0!dropout_16/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<_
dropout_16/dropout/ShapeShapelstm_12/transpose_1:y:0*
T0*
_output_shapes
:§
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype0f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ì
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       È
#batch_normalization_12/moments/meanMeandropout_16/dropout/Mul_1:z:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*#
_output_shapes
:Ð
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferencedropout_16/dropout/Mul_1:z:04batch_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       è
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 £
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 q
,batch_normalization_12/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<±
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0*
T0*
_output_shapes	
:¾
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
&batch_normalization_12/AssignMovingAvgAssignSubVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_12/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ä
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
(batch_normalization_12/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource0batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:·
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¬
&batch_normalization_12/batchnorm/mul_1Muldropout_16/dropout/Mul_1:z:0(batch_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<®
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
:¥
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¶
$batch_normalization_12/batchnorm/subSub7batch_normalization_12/batchnorm/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¼
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<g
lstm_13/ShapeShape*batch_normalization_12/batchnorm/add_1:z:0*
T0*
_output_shapes
:e
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¢
lstm_13/transpose	Transpose*batch_normalization_12/batchnorm/add_1:z:0lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿT
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:g
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask 
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0®
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_13_while_body_166273*%
condR
lstm_13_while_cond_166272*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Û
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<c
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_17/dropout/MulMullstm_13/transpose_1:y:0!dropout_17/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<_
dropout_17/dropout/ShapeShapelstm_13/transpose_1:y:0*
T0*
_output_shapes
:§
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype0f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ì
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       È
#batch_normalization_13/moments/meanMeandropout_17/dropout/Mul_1:z:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*#
_output_shapes
:Ð
0batch_normalization_13/moments/SquaredDifferenceSquaredDifferencedropout_17/dropout/Mul_1:z:04batch_normalization_13/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       è
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 £
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 q
,batch_normalization_13/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<±
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0*
T0*
_output_shapes	
:¾
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:05batch_normalization_13/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
&batch_normalization_13/AssignMovingAvgAssignSubVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_13/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ä
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:07batch_normalization_13/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
(batch_normalization_13/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource0batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:·
$batch_normalization_13/batchnorm/addAddV21batch_normalization_13/moments/Squeeze_1:output:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¬
&batch_normalization_13/batchnorm/mul_1Muldropout_17/dropout/Mul_1:z:0(batch_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<®
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes	
:¥
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¶
$batch_normalization_13/batchnorm/subSub7batch_normalization_13/batchnorm/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¼
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<g
lstm_14/ShapeShape*batch_normalization_13/batchnorm/add_1:z:0*
T0*
_output_shapes
:e
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¢
lstm_14/transpose	Transpose*batch_normalization_13/batchnorm/add_1:z:0lstm_14/transpose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿT
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:g
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask 
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0®
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¨
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ù
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_14_while_body_166450*%
condR
lstm_14_while_cond_166449*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Û
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<c
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_18/dropout/MulMul lstm_14/strided_slice_3:output:0!dropout_18/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dropout_18/dropout/ShapeShape lstm_14/strided_slice_3:output:0*
T0*
_output_shapes
:£
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>È
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5batch_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
#batch_normalization_14/moments/meanMeandropout_18/dropout/Mul_1:z:0>batch_normalization_14/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
+batch_normalization_14/moments/StopGradientStopGradient,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes
:	Ì
0batch_normalization_14/moments/SquaredDifferenceSquaredDifferencedropout_18/dropout/Mul_1:z:04batch_normalization_14/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ä
'batch_normalization_14/moments/varianceMean4batch_normalization_14/moments/SquaredDifference:z:0Bbatch_normalization_14/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
&batch_normalization_14/moments/SqueezeSqueeze,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ¢
(batch_normalization_14/moments/Squeeze_1Squeeze0batch_normalization_14/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 q
,batch_normalization_14/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<±
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
*batch_normalization_14/AssignMovingAvg/subSub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_14/moments/Squeeze:output:0*
T0*
_output_shapes	
:¾
*batch_normalization_14/AssignMovingAvg/mulMul.batch_normalization_14/AssignMovingAvg/sub:z:05batch_normalization_14/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
&batch_normalization_14/AssignMovingAvgAssignSubVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_14/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
,batch_normalization_14/AssignMovingAvg_1/subSub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_14/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ä
,batch_normalization_14/AssignMovingAvg_1/mulMul0batch_normalization_14/AssignMovingAvg_1/sub:z:07batch_normalization_14/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
(batch_normalization_14/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource0batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:·
$batch_normalization_14/batchnorm/addAddV21batch_normalization_14/moments/Squeeze_1:output:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:0;batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¨
&batch_normalization_14/batchnorm/mul_1Muldropout_18/dropout/Mul_1:z:0(batch_normalization_14/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
&batch_normalization_14/batchnorm/mul_2Mul/batch_normalization_14/moments/Squeeze:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes	
:¥
/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¶
$batch_normalization_14/batchnorm/subSub7batch_normalization_14/batchnorm/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense_8/MatMulMatMul*batch_normalization_14/batchnorm/add_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_19/dropout/MulMuldense_8/Relu:activations:0!dropout_19/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
dropout_19/dropout/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:¢
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0f
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ç
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_9/MatMulMatMuldropout_19/dropout/Mul_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp'^batch_normalization_12/AssignMovingAvg6^batch_normalization_12/AssignMovingAvg/ReadVariableOp)^batch_normalization_12/AssignMovingAvg_18^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp4^batch_normalization_12/batchnorm/mul/ReadVariableOp'^batch_normalization_13/AssignMovingAvg6^batch_normalization_13/AssignMovingAvg/ReadVariableOp)^batch_normalization_13/AssignMovingAvg_18^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp4^batch_normalization_13/batchnorm/mul/ReadVariableOp'^batch_normalization_14/AssignMovingAvg6^batch_normalization_14/AssignMovingAvg/ReadVariableOp)^batch_normalization_14/AssignMovingAvg_18^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_14/batchnorm/ReadVariableOp4^batch_normalization_14/batchnorm/mul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_12/AssignMovingAvg&batch_normalization_12/AssignMovingAvg2n
5batch_normalization_12/AssignMovingAvg/ReadVariableOp5batch_normalization_12/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_12/AssignMovingAvg_1(batch_normalization_12/AssignMovingAvg_12r
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2P
&batch_normalization_13/AssignMovingAvg&batch_normalization_13/AssignMovingAvg2n
5batch_normalization_13/AssignMovingAvg/ReadVariableOp5batch_normalization_13/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_13/AssignMovingAvg_1(batch_normalization_13/AssignMovingAvg_12r
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_13/batchnorm/ReadVariableOp/batch_normalization_13/batchnorm/ReadVariableOp2j
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2P
&batch_normalization_14/AssignMovingAvg&batch_normalization_14/AssignMovingAvg2n
5batch_normalization_14/AssignMovingAvg/ReadVariableOp5batch_normalization_14/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_14/AssignMovingAvg_1(batch_normalization_14/AssignMovingAvg_12r
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_14/batchnorm/ReadVariableOp/batch_normalization_14/batchnorm/ReadVariableOp2j
3batch_normalization_14/batchnorm/mul/ReadVariableOp3batch_normalization_14/batchnorm/mul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2Z
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2X
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp*lstm_12/lstm_cell_12/MatMul/ReadVariableOp2\
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp2
lstm_12/whilelstm_12/while2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while2Z
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp*lstm_14/lstm_cell_14/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Q
õ
&sequential_4_lstm_13_while_body_162296F
Bsequential_4_lstm_13_while_sequential_4_lstm_13_while_loop_counterL
Hsequential_4_lstm_13_while_sequential_4_lstm_13_while_maximum_iterations*
&sequential_4_lstm_13_while_placeholder,
(sequential_4_lstm_13_while_placeholder_1,
(sequential_4_lstm_13_while_placeholder_2,
(sequential_4_lstm_13_while_placeholder_3E
Asequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1_0
}sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
^
Jsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
X
Isequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	'
#sequential_4_lstm_13_while_identity)
%sequential_4_lstm_13_while_identity_1)
%sequential_4_lstm_13_while_identity_2)
%sequential_4_lstm_13_while_identity_3)
%sequential_4_lstm_13_while_identity_4)
%sequential_4_lstm_13_while_identity_5C
?sequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1
{sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
\
Hsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
V
Gsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	¢>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp
Lsequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
>sequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_13_while_placeholderUsequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0È
=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpHsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ù
.sequential_4/lstm_13/while/lstm_cell_13/MatMulMatMulEsequential_4/lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpJsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0à
0sequential_4/lstm_13/while/lstm_cell_13/MatMul_1MatMul(sequential_4_lstm_13_while_placeholder_2Gsequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+sequential_4/lstm_13/while/lstm_cell_13/addAddV28sequential_4/lstm_13/while/lstm_cell_13/MatMul:product:0:sequential_4/lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpIsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0æ
/sequential_4/lstm_13/while/lstm_cell_13/BiasAddBiasAdd/sequential_4/lstm_13/while/lstm_cell_13/add:z:0Fsequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7sequential_4/lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :²
-sequential_4/lstm_13/while/lstm_cell_13/splitSplit@sequential_4/lstm_13/while/lstm_cell_13/split/split_dim:output:08sequential_4/lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¥
/sequential_4/lstm_13/while/lstm_cell_13/SigmoidSigmoid6sequential_4/lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid6sequential_4/lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
+sequential_4/lstm_13/while/lstm_cell_13/mulMul5sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_1:y:0(sequential_4_lstm_13_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_4/lstm_13/while/lstm_cell_13/ReluRelu6sequential_4/lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
-sequential_4/lstm_13/while/lstm_cell_13/mul_1Mul3sequential_4/lstm_13/while/lstm_cell_13/Sigmoid:y:0:sequential_4/lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
-sequential_4/lstm_13/while/lstm_cell_13/add_1AddV2/sequential_4/lstm_13/while/lstm_cell_13/mul:z:01sequential_4/lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid6sequential_4/lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.sequential_4/lstm_13/while/lstm_cell_13/Relu_1Relu1sequential_4/lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
-sequential_4/lstm_13/while/lstm_cell_13/mul_2Mul5sequential_4/lstm_13/while/lstm_cell_13/Sigmoid_2:y:0<sequential_4/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?sequential_4/lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_13_while_placeholder_1&sequential_4_lstm_13_while_placeholder1sequential_4/lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 sequential_4/lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_4/lstm_13/while/addAddV2&sequential_4_lstm_13_while_placeholder)sequential_4/lstm_13/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_4/lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 sequential_4/lstm_13/while/add_1AddV2Bsequential_4_lstm_13_while_sequential_4_lstm_13_while_loop_counter+sequential_4/lstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 
#sequential_4/lstm_13/while/IdentityIdentity$sequential_4/lstm_13/while/add_1:z:0 ^sequential_4/lstm_13/while/NoOp*
T0*
_output_shapes
: ¾
%sequential_4/lstm_13/while/Identity_1IdentityHsequential_4_lstm_13_while_sequential_4_lstm_13_while_maximum_iterations ^sequential_4/lstm_13/while/NoOp*
T0*
_output_shapes
: 
%sequential_4/lstm_13/while/Identity_2Identity"sequential_4/lstm_13/while/add:z:0 ^sequential_4/lstm_13/while/NoOp*
T0*
_output_shapes
: Ø
%sequential_4/lstm_13/while/Identity_3IdentityOsequential_4/lstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_4/lstm_13/while/NoOp*
T0*
_output_shapes
: :éèÒ¹
%sequential_4/lstm_13/while/Identity_4Identity1sequential_4/lstm_13/while/lstm_cell_13/mul_2:z:0 ^sequential_4/lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%sequential_4/lstm_13/while/Identity_5Identity1sequential_4/lstm_13/while/lstm_cell_13/add_1:z:0 ^sequential_4/lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
sequential_4/lstm_13/while/NoOpNoOp?^sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_4_lstm_13_while_identity,sequential_4/lstm_13/while/Identity:output:0"W
%sequential_4_lstm_13_while_identity_1.sequential_4/lstm_13/while/Identity_1:output:0"W
%sequential_4_lstm_13_while_identity_2.sequential_4/lstm_13/while/Identity_2:output:0"W
%sequential_4_lstm_13_while_identity_3.sequential_4/lstm_13/while/Identity_3:output:0"W
%sequential_4_lstm_13_while_identity_4.sequential_4/lstm_13/while/Identity_4:output:0"W
%sequential_4_lstm_13_while_identity_5.sequential_4/lstm_13/while/Identity_5:output:0"
Gsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resourceIsequential_4_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"
Hsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceJsequential_4_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"
Fsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceHsequential_4_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"
?sequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1Asequential_4_lstm_13_while_sequential_4_lstm_13_strided_slice_1_0"ü
{sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>sequential_4/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2~
=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp=sequential_4/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2
?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?sequential_4/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
#
å
while_body_162840
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_12_162864_0:	/
while_lstm_cell_12_162866_0:
*
while_lstm_cell_12_162868_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_12_162864:	-
while_lstm_cell_12_162866:
(
while_lstm_cell_12_162868:	¢*while/lstm_cell_12/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¶
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_162864_0while_lstm_cell_12_162866_0while_lstm_cell_12_162868_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162781Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_12_162864while_lstm_cell_12_162864_0"8
while_lstm_cell_12_162866while_lstm_cell_12_162866_0"8
while_lstm_cell_12_162868while_lstm_cell_12_162868_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
 ¥
ã1
"__inference__traced_restore_169646
file_prefix<
-assignvariableop_batch_normalization_12_gamma:	=
.assignvariableop_1_batch_normalization_12_beta:	D
5assignvariableop_2_batch_normalization_12_moving_mean:	H
9assignvariableop_3_batch_normalization_12_moving_variance:	>
/assignvariableop_4_batch_normalization_13_gamma:	=
.assignvariableop_5_batch_normalization_13_beta:	D
5assignvariableop_6_batch_normalization_13_moving_mean:	H
9assignvariableop_7_batch_normalization_13_moving_variance:	>
/assignvariableop_8_batch_normalization_14_gamma:	=
.assignvariableop_9_batch_normalization_14_beta:	E
6assignvariableop_10_batch_normalization_14_moving_mean:	I
:assignvariableop_11_batch_normalization_14_moving_variance:	5
"assignvariableop_12_dense_8_kernel:	 .
 assignvariableop_13_dense_8_bias: 4
"assignvariableop_14_dense_9_kernel: .
 assignvariableop_15_dense_9_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: B
/assignvariableop_21_lstm_12_lstm_cell_12_kernel:	M
9assignvariableop_22_lstm_12_lstm_cell_12_recurrent_kernel:
<
-assignvariableop_23_lstm_12_lstm_cell_12_bias:	C
/assignvariableop_24_lstm_13_lstm_cell_13_kernel:
M
9assignvariableop_25_lstm_13_lstm_cell_13_recurrent_kernel:
<
-assignvariableop_26_lstm_13_lstm_cell_13_bias:	C
/assignvariableop_27_lstm_14_lstm_cell_14_kernel:
M
9assignvariableop_28_lstm_14_lstm_cell_14_recurrent_kernel:
<
-assignvariableop_29_lstm_14_lstm_cell_14_bias:	#
assignvariableop_30_total: #
assignvariableop_31_count: %
assignvariableop_32_total_1: %
assignvariableop_33_count_1: F
7assignvariableop_34_adam_batch_normalization_12_gamma_m:	E
6assignvariableop_35_adam_batch_normalization_12_beta_m:	F
7assignvariableop_36_adam_batch_normalization_13_gamma_m:	E
6assignvariableop_37_adam_batch_normalization_13_beta_m:	F
7assignvariableop_38_adam_batch_normalization_14_gamma_m:	E
6assignvariableop_39_adam_batch_normalization_14_beta_m:	<
)assignvariableop_40_adam_dense_8_kernel_m:	 5
'assignvariableop_41_adam_dense_8_bias_m: ;
)assignvariableop_42_adam_dense_9_kernel_m: 5
'assignvariableop_43_adam_dense_9_bias_m:I
6assignvariableop_44_adam_lstm_12_lstm_cell_12_kernel_m:	T
@assignvariableop_45_adam_lstm_12_lstm_cell_12_recurrent_kernel_m:
C
4assignvariableop_46_adam_lstm_12_lstm_cell_12_bias_m:	J
6assignvariableop_47_adam_lstm_13_lstm_cell_13_kernel_m:
T
@assignvariableop_48_adam_lstm_13_lstm_cell_13_recurrent_kernel_m:
C
4assignvariableop_49_adam_lstm_13_lstm_cell_13_bias_m:	J
6assignvariableop_50_adam_lstm_14_lstm_cell_14_kernel_m:
T
@assignvariableop_51_adam_lstm_14_lstm_cell_14_recurrent_kernel_m:
C
4assignvariableop_52_adam_lstm_14_lstm_cell_14_bias_m:	F
7assignvariableop_53_adam_batch_normalization_12_gamma_v:	E
6assignvariableop_54_adam_batch_normalization_12_beta_v:	F
7assignvariableop_55_adam_batch_normalization_13_gamma_v:	E
6assignvariableop_56_adam_batch_normalization_13_beta_v:	F
7assignvariableop_57_adam_batch_normalization_14_gamma_v:	E
6assignvariableop_58_adam_batch_normalization_14_beta_v:	<
)assignvariableop_59_adam_dense_8_kernel_v:	 5
'assignvariableop_60_adam_dense_8_bias_v: ;
)assignvariableop_61_adam_dense_9_kernel_v: 5
'assignvariableop_62_adam_dense_9_bias_v:I
6assignvariableop_63_adam_lstm_12_lstm_cell_12_kernel_v:	T
@assignvariableop_64_adam_lstm_12_lstm_cell_12_recurrent_kernel_v:
C
4assignvariableop_65_adam_lstm_12_lstm_cell_12_bias_v:	J
6assignvariableop_66_adam_lstm_13_lstm_cell_13_kernel_v:
T
@assignvariableop_67_adam_lstm_13_lstm_cell_13_recurrent_kernel_v:
C
4assignvariableop_68_adam_lstm_13_lstm_cell_13_bias_v:	J
6assignvariableop_69_adam_lstm_14_lstm_cell_14_kernel_v:
T
@assignvariableop_70_adam_lstm_14_lstm_cell_14_recurrent_kernel_v:
C
4assignvariableop_71_adam_lstm_14_lstm_cell_14_bias_v:	
identity_73¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_8¢AssignVariableOp_9%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*¼$
value²$B¯$IB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*§
valueBIB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*W
dtypesM
K2I	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp-assignvariableop_batch_normalization_12_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_12_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_2AssignVariableOp5assignvariableop_2_batch_normalization_12_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_normalization_12_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_13_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_13_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_13_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_13_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_14_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_14_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_14_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_14_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_8_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_8_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_9_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_9_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_21AssignVariableOp/assignvariableop_21_lstm_12_lstm_cell_12_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_22AssignVariableOp9assignvariableop_22_lstm_12_lstm_cell_12_recurrent_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp-assignvariableop_23_lstm_12_lstm_cell_12_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_24AssignVariableOp/assignvariableop_24_lstm_13_lstm_cell_13_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_25AssignVariableOp9assignvariableop_25_lstm_13_lstm_cell_13_recurrent_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp-assignvariableop_26_lstm_13_lstm_cell_13_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_27AssignVariableOp/assignvariableop_27_lstm_14_lstm_cell_14_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_28AssignVariableOp9assignvariableop_28_lstm_14_lstm_cell_14_recurrent_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp-assignvariableop_29_lstm_14_lstm_cell_14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_batch_normalization_12_gamma_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_12_beta_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adam_batch_normalization_13_gamma_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_batch_normalization_13_beta_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adam_batch_normalization_14_gamma_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_14_beta_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_8_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_8_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_9_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_9_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_lstm_12_lstm_cell_12_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_45AssignVariableOp@assignvariableop_45_adam_lstm_12_lstm_cell_12_recurrent_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_lstm_12_lstm_cell_12_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_lstm_13_lstm_cell_13_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_48AssignVariableOp@assignvariableop_48_adam_lstm_13_lstm_cell_13_recurrent_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_lstm_13_lstm_cell_13_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_lstm_14_lstm_cell_14_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_51AssignVariableOp@assignvariableop_51_adam_lstm_14_lstm_cell_14_recurrent_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_lstm_14_lstm_cell_14_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_12_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_12_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_13_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_13_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_14_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_14_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_8_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_8_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_9_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_9_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_lstm_12_lstm_cell_12_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_64AssignVariableOp@assignvariableop_64_adam_lstm_12_lstm_cell_12_recurrent_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_65AssignVariableOp4assignvariableop_65_adam_lstm_12_lstm_cell_12_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_lstm_13_lstm_cell_13_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_67AssignVariableOp@assignvariableop_67_adam_lstm_13_lstm_cell_13_recurrent_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_68AssignVariableOp4assignvariableop_68_adam_lstm_13_lstm_cell_13_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_lstm_14_lstm_cell_14_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_70AssignVariableOp@assignvariableop_70_adam_lstm_14_lstm_cell_14_recurrent_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_71AssignVariableOp4assignvariableop_71_adam_lstm_14_lstm_cell_14_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ÿ
Identity_72Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_73IdentityIdentity_72:output:0^NoOp_1*
T0*
_output_shapes
: ì
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_73Identity_73:output:0*§
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


e
F__inference_dropout_16_layer_call_and_return_conditional_losses_167294

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
#
ç
while_body_163272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_13_163296_0:
/
while_lstm_cell_13_163298_0:
*
while_lstm_cell_13_163300_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_13_163296:
-
while_lstm_cell_13_163298:
(
while_lstm_cell_13_163300:	¢*while/lstm_cell_13/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¶
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_163296_0while_lstm_cell_13_163298_0while_lstm_cell_13_163300_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163213Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_13_163296while_lstm_cell_13_163296_0"8
while_lstm_cell_13_163298while_lstm_cell_13_163298_0"8
while_lstm_cell_13_163300while_lstm_cell_13_163300_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹8
Ô
while_body_168629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_14_matmul_readvariableop_resource_0:
I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_14_matmul_readvariableop_resource:
G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
A
2while_lstm_cell_14_biasadd_readvariableop_resource:	¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

d
+__inference_dropout_17_layer_call_fn_168000

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_164715t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
ä
Ö
7__inference_batch_normalization_13_layer_call_fn_168043

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163421}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_169149

inputs
states_0
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
°%
ï
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163853

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
&
ï
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163421

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       £
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ã
lstm_14_while_cond_166449,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1D
@lstm_14_while_lstm_14_while_cond_166449___redundant_placeholder0D
@lstm_14_while_lstm_14_while_cond_166449___redundant_placeholder1D
@lstm_14_while_lstm_14_while_cond_166449___redundant_placeholder2D
@lstm_14_while_lstm_14_while_cond_166449___redundant_placeholder3
lstm_14_while_identity

lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: [
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_14_while_identitylstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÄJ

C__inference_lstm_13_layer_call_and_return_conditional_losses_164179

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
A
-lstm_cell_13_matmul_1_readvariableop_resource:
;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164095*
condR
while_cond_164094*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¢8

C__inference_lstm_12_layer_call_and_return_conditional_losses_162718

inputs&
lstm_cell_12_162636:	'
lstm_cell_12_162638:
"
lstm_cell_12_162640:	
identity¢$lstm_cell_12/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskø
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_162636lstm_cell_12_162638lstm_cell_12_162640*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162635n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_162636lstm_cell_12_162638lstm_cell_12_162640*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_162649*
condR
while_cond_162648*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ã
while_cond_162839
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_162839___redundant_placeholder04
0while_while_cond_162839___redundant_placeholder14
0while_while_cond_162839___redundant_placeholder24
0while_while_cond_162839___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

¶
(__inference_lstm_12_layer_call_fn_166695

inputs
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_165062t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹
Ã
while_cond_168628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_168628___redundant_placeholder04
0while_while_cond_168628___redundant_placeholder14
0while_while_cond_168628___redundant_placeholder24
0while_while_cond_168628___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ö
÷
-__inference_lstm_cell_12_layer_call_fn_168904

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162635p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1


ã
lstm_12_while_cond_165608,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1D
@lstm_12_while_lstm_12_while_cond_165608___redundant_placeholder0D
@lstm_12_while_lstm_12_while_cond_165608___redundant_placeholder1D
@lstm_12_while_lstm_12_while_cond_165608___redundant_placeholder2D
@lstm_12_while_lstm_12_while_cond_165608___redundant_placeholder3
lstm_12_while_identity

lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: [
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_12_while_identitylstm_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
&
ï
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_168097

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       £
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_169083

inputs
states_0
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¬
¸
(__inference_lstm_12_layer_call_fn_166662
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_162718}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ù
ø
-__inference_lstm_cell_13_layer_call_fn_169019

inputs
states_0
states_1
unknown:

	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163213p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
òA
Ô

lstm_13_while_body_165765,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
Q
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
K
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorM
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
O
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
I
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	¢1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0®
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ò
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: q
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: ±
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_13/while/NoOpNoOp2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"È
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¹
(__inference_lstm_14_layer_call_fn_168119
inputs_0
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_163773p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
òA
Ô

lstm_13_while_body_166273,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
Q
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
K
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorM
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
O
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
I
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	¢1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0®
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ò
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: q
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: ±
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_13/while/NoOpNoOp2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"È
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¬
¸
(__inference_lstm_12_layer_call_fn_166673
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_162909}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ì
¥
-__inference_sequential_4_layer_call_fn_165302
lstm_12_input
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:

	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:


unknown_14:


unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	 

unknown_21: 

unknown_22: 

unknown_23:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_165194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
'
_user_specified_namelstm_12_input
#
ç
while_body_163081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_13_163105_0:
/
while_lstm_cell_13_163107_0:
*
while_lstm_cell_13_163109_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_13_163105:
-
while_lstm_cell_13_163107:
(
while_lstm_cell_13_163109:	¢*while/lstm_cell_13/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¶
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_163105_0while_lstm_cell_13_163107_0while_lstm_cell_13_163109_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163067Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_13_163105while_lstm_cell_13_163105_0"8
while_lstm_cell_13_163107while_lstm_cell_13_163107_0"8
while_lstm_cell_13_163109while_lstm_cell_13_163109_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹8
Ô
while_body_164095
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
A
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ý
·
(__inference_lstm_14_layer_call_fn_168130

inputs
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_164345p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹
Ã
while_cond_167619
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167619___redundant_placeholder04
0while_while_cond_167619___redundant_placeholder14
0while_while_cond_167619___redundant_placeholder24
0while_while_cond_167619___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹
Ã
while_cond_168485
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_168485___redundant_placeholder04
0while_while_cond_168485___redundant_placeholder14
0while_while_cond_168485___redundant_placeholder24
0while_while_cond_168485___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ý
·
(__inference_lstm_14_layer_call_fn_168141

inputs
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_164686p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
ûC

H__inference_sequential_4_layer_call_and_return_conditional_losses_165434
lstm_12_input!
lstm_12_165371:	"
lstm_12_165373:

lstm_12_165375:	,
batch_normalization_12_165379:	,
batch_normalization_12_165381:	,
batch_normalization_12_165383:	,
batch_normalization_12_165385:	"
lstm_13_165388:
"
lstm_13_165390:

lstm_13_165392:	,
batch_normalization_13_165396:	,
batch_normalization_13_165398:	,
batch_normalization_13_165400:	,
batch_normalization_13_165402:	"
lstm_14_165405:
"
lstm_14_165407:

lstm_14_165409:	,
batch_normalization_14_165413:	,
batch_normalization_14_165415:	,
batch_normalization_14_165417:	,
batch_normalization_14_165419:	!
dense_8_165422:	 
dense_8_165424:  
dense_9_165428: 
dense_9_165430:
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢.batch_normalization_14/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢"dropout_18/StatefulPartitionedCall¢"dropout_19/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_165371lstm_12_165373lstm_12_165375*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_165062ó
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_164903
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0batch_normalization_12_165379batch_normalization_12_165381batch_normalization_12_165383batch_normalization_12_165385*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162989´
lstm_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0lstm_13_165388lstm_13_165390lstm_13_165392*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_164874
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_164715
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0batch_normalization_13_165396batch_normalization_13_165398batch_normalization_13_165400batch_normalization_13_165402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163421°
lstm_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0lstm_14_165405lstm_14_165407lstm_14_165409*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_164686
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_164527
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0batch_normalization_14_165413batch_normalization_14_165415batch_normalization_14_165417batch_normalization_14_165419*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163853
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_8_165422dense_8_165424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_164380
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_164494
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_165428dense_9_165430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_164404w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
'
_user_specified_namelstm_12_input
Q
ó
&sequential_4_lstm_12_while_body_162140F
Bsequential_4_lstm_12_while_sequential_4_lstm_12_while_loop_counterL
Hsequential_4_lstm_12_while_sequential_4_lstm_12_while_maximum_iterations*
&sequential_4_lstm_12_while_placeholder,
(sequential_4_lstm_12_while_placeholder_1,
(sequential_4_lstm_12_while_placeholder_2,
(sequential_4_lstm_12_while_placeholder_3E
Asequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1_0
}sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	^
Jsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
X
Isequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	'
#sequential_4_lstm_12_while_identity)
%sequential_4_lstm_12_while_identity_1)
%sequential_4_lstm_12_while_identity_2)
%sequential_4_lstm_12_while_identity_3)
%sequential_4_lstm_12_while_identity_4)
%sequential_4_lstm_12_while_identity_5C
?sequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1
{sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensorY
Fsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	\
Hsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
V
Gsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	¢>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp¢=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp¢?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp
Lsequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
>sequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_12_while_placeholderUsequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ç
=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOpHsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ù
.sequential_4/lstm_12/while/lstm_cell_12/MatMulMatMulEsequential_4/lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpJsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0à
0sequential_4/lstm_12/while/lstm_cell_12/MatMul_1MatMul(sequential_4_lstm_12_while_placeholder_2Gsequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+sequential_4/lstm_12/while/lstm_cell_12/addAddV28sequential_4/lstm_12/while/lstm_cell_12/MatMul:product:0:sequential_4/lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpIsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0æ
/sequential_4/lstm_12/while/lstm_cell_12/BiasAddBiasAdd/sequential_4/lstm_12/while/lstm_cell_12/add:z:0Fsequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7sequential_4/lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :²
-sequential_4/lstm_12/while/lstm_cell_12/splitSplit@sequential_4/lstm_12/while/lstm_cell_12/split/split_dim:output:08sequential_4/lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¥
/sequential_4/lstm_12/while/lstm_cell_12/SigmoidSigmoid6sequential_4/lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid6sequential_4/lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
+sequential_4/lstm_12/while/lstm_cell_12/mulMul5sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_1:y:0(sequential_4_lstm_12_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_4/lstm_12/while/lstm_cell_12/ReluRelu6sequential_4/lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
-sequential_4/lstm_12/while/lstm_cell_12/mul_1Mul3sequential_4/lstm_12/while/lstm_cell_12/Sigmoid:y:0:sequential_4/lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
-sequential_4/lstm_12/while/lstm_cell_12/add_1AddV2/sequential_4/lstm_12/while/lstm_cell_12/mul:z:01sequential_4/lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid6sequential_4/lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.sequential_4/lstm_12/while/lstm_cell_12/Relu_1Relu1sequential_4/lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
-sequential_4/lstm_12/while/lstm_cell_12/mul_2Mul5sequential_4/lstm_12/while/lstm_cell_12/Sigmoid_2:y:0<sequential_4/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?sequential_4/lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_12_while_placeholder_1&sequential_4_lstm_12_while_placeholder1sequential_4/lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 sequential_4/lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_4/lstm_12/while/addAddV2&sequential_4_lstm_12_while_placeholder)sequential_4/lstm_12/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_4/lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 sequential_4/lstm_12/while/add_1AddV2Bsequential_4_lstm_12_while_sequential_4_lstm_12_while_loop_counter+sequential_4/lstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 
#sequential_4/lstm_12/while/IdentityIdentity$sequential_4/lstm_12/while/add_1:z:0 ^sequential_4/lstm_12/while/NoOp*
T0*
_output_shapes
: ¾
%sequential_4/lstm_12/while/Identity_1IdentityHsequential_4_lstm_12_while_sequential_4_lstm_12_while_maximum_iterations ^sequential_4/lstm_12/while/NoOp*
T0*
_output_shapes
: 
%sequential_4/lstm_12/while/Identity_2Identity"sequential_4/lstm_12/while/add:z:0 ^sequential_4/lstm_12/while/NoOp*
T0*
_output_shapes
: Ø
%sequential_4/lstm_12/while/Identity_3IdentityOsequential_4/lstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_4/lstm_12/while/NoOp*
T0*
_output_shapes
: :éèÒ¹
%sequential_4/lstm_12/while/Identity_4Identity1sequential_4/lstm_12/while/lstm_cell_12/mul_2:z:0 ^sequential_4/lstm_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%sequential_4/lstm_12/while/Identity_5Identity1sequential_4/lstm_12/while/lstm_cell_12/add_1:z:0 ^sequential_4/lstm_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
sequential_4/lstm_12/while/NoOpNoOp?^sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_4_lstm_12_while_identity,sequential_4/lstm_12/while/Identity:output:0"W
%sequential_4_lstm_12_while_identity_1.sequential_4/lstm_12/while/Identity_1:output:0"W
%sequential_4_lstm_12_while_identity_2.sequential_4/lstm_12/while/Identity_2:output:0"W
%sequential_4_lstm_12_while_identity_3.sequential_4/lstm_12/while/Identity_3:output:0"W
%sequential_4_lstm_12_while_identity_4.sequential_4/lstm_12/while/Identity_4:output:0"W
%sequential_4_lstm_12_while_identity_5.sequential_4/lstm_12/while/Identity_5:output:0"
Gsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resourceIsequential_4_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"
Hsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceJsequential_4_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"
Fsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceHsequential_4_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"
?sequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1Asequential_4_lstm_12_while_sequential_4_lstm_12_strided_slice_1_0"ü
{sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>sequential_4/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2~
=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp=sequential_4/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2
?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?sequential_4/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
üJ

C__inference_lstm_12_layer_call_and_return_conditional_losses_166838
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	A
-lstm_cell_12_matmul_1_readvariableop_resource:
;
,lstm_cell_12_biasadd_readvariableop_resource:	
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_166754*
condR
while_cond_166753*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ï

H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163067

inputs

states
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates


e
F__inference_dropout_16_layer_call_and_return_conditional_losses_164903

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹
Ã
while_cond_167905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167905___redundant_placeholder04
0while_while_cond_167905___redundant_placeholder14
0while_while_cond_167905___redundant_placeholder24
0while_while_cond_167905___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

ç
&sequential_4_lstm_13_while_cond_162295F
Bsequential_4_lstm_13_while_sequential_4_lstm_13_while_loop_counterL
Hsequential_4_lstm_13_while_sequential_4_lstm_13_while_maximum_iterations*
&sequential_4_lstm_13_while_placeholder,
(sequential_4_lstm_13_while_placeholder_1,
(sequential_4_lstm_13_while_placeholder_2,
(sequential_4_lstm_13_while_placeholder_3H
Dsequential_4_lstm_13_while_less_sequential_4_lstm_13_strided_slice_1^
Zsequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_162295___redundant_placeholder0^
Zsequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_162295___redundant_placeholder1^
Zsequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_162295___redundant_placeholder2^
Zsequential_4_lstm_13_while_sequential_4_lstm_13_while_cond_162295___redundant_placeholder3'
#sequential_4_lstm_13_while_identity
¶
sequential_4/lstm_13/while/LessLess&sequential_4_lstm_13_while_placeholderDsequential_4_lstm_13_while_less_sequential_4_lstm_13_strided_slice_1*
T0*
_output_shapes
: u
#sequential_4/lstm_13/while/IdentityIdentity#sequential_4/lstm_13/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_4_lstm_13_while_identity,sequential_4/lstm_13/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
§8

C__inference_lstm_13_layer_call_and_return_conditional_losses_163341

inputs'
lstm_cell_13_163259:
'
lstm_cell_13_163261:
"
lstm_cell_13_163263:	
identity¢$lstm_cell_13/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskø
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_163259lstm_cell_13_163261lstm_cell_13_163263*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163213n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_163259lstm_cell_13_163261lstm_cell_13_163263*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163272*
condR
while_cond_163271*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_164527

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
ø
-__inference_lstm_cell_14_layer_call_fn_169117

inputs
states_0
states_1
unknown:

	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163645p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¹
Ã
while_cond_167762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167762___redundant_placeholder04
0while_while_cond_167762___redundant_placeholder14
0while_while_cond_167762___redundant_placeholder24
0while_while_cond_167762___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
#
ç
while_body_163704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_14_163728_0:
/
while_lstm_cell_14_163730_0:
*
while_lstm_cell_14_163732_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_14_163728:
-
while_lstm_cell_14_163730:
(
while_lstm_cell_14_163732:	¢*while/lstm_cell_14/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¶
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_163728_0while_lstm_cell_14_163730_0while_lstm_cell_14_163732_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163645Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_14_163728while_lstm_cell_14_163728_0"8
while_lstm_cell_14_163730while_lstm_cell_14_163730_0"8
while_lstm_cell_14_163732while_lstm_cell_14_163732_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


ô
C__inference_dense_9_layer_call_and_return_conditional_losses_164404

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó

H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_168985

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1


ã
lstm_13_while_cond_165764,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1D
@lstm_13_while_lstm_13_while_cond_165764___redundant_placeholder0D
@lstm_13_while_lstm_13_while_cond_165764___redundant_placeholder1D
@lstm_13_while_lstm_13_while_cond_165764___redundant_placeholder2D
@lstm_13_while_lstm_13_while_cond_165764___redundant_placeholder3
lstm_13_while_identity

lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: [
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_13_while_identitylstm_13/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹8
Ô
while_body_168486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_14_matmul_readvariableop_resource_0:
I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_14_matmul_readvariableop_resource:
G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
A
2while_lstm_cell_14_biasadd_readvariableop_resource:	¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_168855

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ö
÷
-__inference_lstm_cell_12_layer_call_fn_168921

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162781p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
²
Ö
7__inference_batch_normalization_14_layer_call_fn_168753

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163806p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾J

C__inference_lstm_12_layer_call_and_return_conditional_losses_164013

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	A
-lstm_cell_12_matmul_1_readvariableop_resource:
;
,lstm_cell_12_biasadd_readvariableop_resource:	
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163929*
condR
while_cond_163928*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹
Ã
while_cond_164977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_164977___redundant_placeholder04
0while_while_cond_164977___redundant_placeholder14
0while_while_cond_164977___redundant_placeholder24
0while_while_cond_164977___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹
Ã
while_cond_162648
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_162648___redundant_placeholder04
0while_while_cond_162648___redundant_placeholder14
0while_while_cond_162648___redundant_placeholder24
0while_while_cond_162648___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
æ
Ö
7__inference_batch_normalization_12_layer_call_fn_167307

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162942}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_168867

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_164391

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ù
ø
-__inference_lstm_cell_13_layer_call_fn_169002

inputs
states_0
states_1
unknown:

	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163067p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¢

$__inference_signature_wrapper_166651
lstm_12_input
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:

	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:


unknown_14:


unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	 

unknown_21: 

unknown_22: 

unknown_23:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_162568o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
'
_user_specified_namelstm_12_input
ÉJ

C__inference_lstm_14_layer_call_and_return_conditional_losses_168570

inputs?
+lstm_cell_14_matmul_readvariableop_resource:
A
-lstm_cell_14_matmul_1_readvariableop_resource:
;
,lstm_cell_14_biasadd_readvariableop_resource:	
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_168486*
condR
while_cond_168485*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¾J

C__inference_lstm_12_layer_call_and_return_conditional_losses_167124

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	A
-lstm_cell_12_matmul_1_readvariableop_resource:
;
,lstm_cell_12_biasadd_readvariableop_resource:	
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_167040*
condR
while_cond_167039*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
&
ï
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162989

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       £
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·

-__inference_sequential_4_layer_call_fn_165550

inputs
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:

	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:


unknown_14:


unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	 

unknown_21: 

unknown_22: 

unknown_23:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_165194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹8
Ô
while_body_167477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
A
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ë

H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162781

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¹
Ã
while_cond_168342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_168342___redundant_placeholder04
0while_while_cond_168342___redundant_placeholder14
0while_while_cond_168342___redundant_placeholder24
0while_while_cond_168342___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

¹
(__inference_lstm_14_layer_call_fn_168108
inputs_0
unknown:

	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_163582p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
#
å
while_body_162649
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_12_162673_0:	/
while_lstm_cell_12_162675_0:
*
while_lstm_cell_12_162677_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_12_162673:	-
while_lstm_cell_12_162675:
(
while_lstm_cell_12_162677:	¢*while/lstm_cell_12/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¶
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_162673_0while_lstm_cell_12_162675_0while_lstm_cell_12_162677_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162635Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_12_162673while_lstm_cell_12_162673_0"8
while_lstm_cell_12_162675while_lstm_cell_12_162675_0"8
while_lstm_cell_12_162677while_lstm_cell_12_162677_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹
Ã
while_cond_164260
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_164260___redundant_placeholder04
0while_while_cond_164260___redundant_placeholder14
0while_while_cond_164260___redundant_placeholder24
0while_while_cond_164260___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ó

H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_168953

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ô	
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_164494

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
á
µ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_163806

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_163213

inputs

states
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates

d
+__inference_dropout_16_layer_call_fn_167277

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_164903t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
#
ç
while_body_163513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_14_163537_0:
/
while_lstm_cell_14_163539_0:
*
while_lstm_cell_14_163541_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_14_163537:
-
while_lstm_cell_14_163539:
(
while_lstm_cell_14_163541:	¢*while/lstm_cell_14/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¶
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_163537_0while_lstm_cell_14_163539_0while_lstm_cell_14_163541_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163499Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_14_163537while_lstm_cell_14_163537_0"8
while_lstm_cell_14_163539while_lstm_cell_14_163539_0"8
while_lstm_cell_14_163541while_lstm_cell_14_163541_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
þJ

C__inference_lstm_14_layer_call_and_return_conditional_losses_168427
inputs_0?
+lstm_cell_14_matmul_readvariableop_resource:
A
-lstm_cell_14_matmul_1_readvariableop_resource:
;
,lstm_cell_14_biasadd_readvariableop_resource:	
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_168343*
condR
while_cond_168342*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
µ8
Ò
while_body_166754
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
A
2while_lstm_cell_12_biasadd_readvariableop_resource:	¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0º
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹
Ã
while_cond_164094
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_164094___redundant_placeholder04
0while_while_cond_164094___redundant_placeholder14
0while_while_cond_164094___redundant_placeholder24
0while_while_cond_164094___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


ã
lstm_12_while_cond_166095,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1D
@lstm_12_while_lstm_12_while_cond_166095___redundant_placeholder0D
@lstm_12_while_lstm_12_while_cond_166095___redundant_placeholder1D
@lstm_12_while_lstm_12_while_cond_166095___redundant_placeholder2D
@lstm_12_while_lstm_12_while_cond_166095___redundant_placeholder3
lstm_12_while_identity

lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: [
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_12_while_identitylstm_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹8
Ô
while_body_168343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_14_matmul_readvariableop_resource_0:
I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_14_matmul_readvariableop_resource:
G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
A
2while_lstm_cell_14_biasadd_readvariableop_resource:	¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹8
Ô
while_body_164602
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_14_matmul_readvariableop_resource_0:
I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_14_matmul_readvariableop_resource:
G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
A
2while_lstm_cell_14_biasadd_readvariableop_resource:	¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÄJ

C__inference_lstm_13_layer_call_and_return_conditional_losses_167990

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
A
-lstm_cell_13_matmul_1_readvariableop_resource:
;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_167906*
condR
while_cond_167905*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
÷
d
+__inference_dropout_18_layer_call_fn_168723

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_164527p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ô
C__inference_dense_9_layer_call_and_return_conditional_losses_168887

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÉJ

C__inference_lstm_14_layer_call_and_return_conditional_losses_164686

inputs?
+lstm_cell_14_matmul_readvariableop_resource:
A
-lstm_cell_14_matmul_1_readvariableop_resource:
;
,lstm_cell_14_biasadd_readvariableop_resource:	
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164602*
condR
while_cond_164601*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
÷

H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_169181

inputs
states_0
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¹
Ã
while_cond_167182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167182___redundant_placeholder04
0while_while_cond_167182___redundant_placeholder14
0while_while_cond_167182___redundant_placeholder24
0while_while_cond_167182___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾J

C__inference_lstm_12_layer_call_and_return_conditional_losses_165062

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	A
-lstm_cell_12_matmul_1_readvariableop_resource:
;
,lstm_cell_12_biasadd_readvariableop_resource:	
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_164978*
condR
while_cond_164977*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹
Ã
while_cond_163703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163703___redundant_placeholder04
0while_while_cond_163703___redundant_placeholder14
0while_while_cond_163703___redundant_placeholder24
0while_while_cond_163703___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

ç
&sequential_4_lstm_12_while_cond_162139F
Bsequential_4_lstm_12_while_sequential_4_lstm_12_while_loop_counterL
Hsequential_4_lstm_12_while_sequential_4_lstm_12_while_maximum_iterations*
&sequential_4_lstm_12_while_placeholder,
(sequential_4_lstm_12_while_placeholder_1,
(sequential_4_lstm_12_while_placeholder_2,
(sequential_4_lstm_12_while_placeholder_3H
Dsequential_4_lstm_12_while_less_sequential_4_lstm_12_strided_slice_1^
Zsequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_162139___redundant_placeholder0^
Zsequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_162139___redundant_placeholder1^
Zsequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_162139___redundant_placeholder2^
Zsequential_4_lstm_12_while_sequential_4_lstm_12_while_cond_162139___redundant_placeholder3'
#sequential_4_lstm_12_while_identity
¶
sequential_4/lstm_12/while/LessLess&sequential_4_lstm_12_while_placeholderDsequential_4_lstm_12_while_less_sequential_4_lstm_12_strided_slice_1*
T0*
_output_shapes
: u
#sequential_4/lstm_12/while/IdentityIdentity#sequential_4/lstm_12/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_4_lstm_12_while_identity,sequential_4/lstm_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹
Ã
while_cond_167039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167039___redundant_placeholder04
0while_while_cond_167039___redundant_placeholder14
0while_while_cond_167039___redundant_placeholder24
0while_while_cond_167039___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ó
d
+__inference_dropout_19_layer_call_fn_168850

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_164494o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¹
Ã
while_cond_164789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_164789___redundant_placeholder04
0while_while_cond_164789___redundant_placeholder14
0while_while_cond_164789___redundant_placeholder24
0while_while_cond_164789___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¥
G
+__inference_dropout_18_layer_call_fn_168718

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_164358a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_168740

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
¥
-__inference_sequential_4_layer_call_fn_164464
lstm_12_input
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:

	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:


unknown_14:


unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	 

unknown_21: 

unknown_22: 

unknown_23:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_164411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
'
_user_specified_namelstm_12_input
¹8
Ô
while_body_167620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
A
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
æ
Ö
7__inference_batch_normalization_13_layer_call_fn_168030

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_163374}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°%
ï
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_168820

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ã
while_cond_166896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_166896___redundant_placeholder04
0while_while_cond_166896___redundant_placeholder14
0while_while_cond_166896___redundant_placeholder24
0while_while_cond_166896___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

©
!__inference__wrapped_model_162568
lstm_12_inputS
@sequential_4_lstm_12_lstm_cell_12_matmul_readvariableop_resource:	V
Bsequential_4_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
P
Asequential_4_lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	T
Esequential_4_batch_normalization_12_batchnorm_readvariableop_resource:	X
Isequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resource:	V
Gsequential_4_batch_normalization_12_batchnorm_readvariableop_1_resource:	V
Gsequential_4_batch_normalization_12_batchnorm_readvariableop_2_resource:	T
@sequential_4_lstm_13_lstm_cell_13_matmul_readvariableop_resource:
V
Bsequential_4_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
P
Asequential_4_lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	T
Esequential_4_batch_normalization_13_batchnorm_readvariableop_resource:	X
Isequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resource:	V
Gsequential_4_batch_normalization_13_batchnorm_readvariableop_1_resource:	V
Gsequential_4_batch_normalization_13_batchnorm_readvariableop_2_resource:	T
@sequential_4_lstm_14_lstm_cell_14_matmul_readvariableop_resource:
V
Bsequential_4_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
P
Asequential_4_lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	T
Esequential_4_batch_normalization_14_batchnorm_readvariableop_resource:	X
Isequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resource:	V
Gsequential_4_batch_normalization_14_batchnorm_readvariableop_1_resource:	V
Gsequential_4_batch_normalization_14_batchnorm_readvariableop_2_resource:	F
3sequential_4_dense_8_matmul_readvariableop_resource:	 B
4sequential_4_dense_8_biasadd_readvariableop_resource: E
3sequential_4_dense_9_matmul_readvariableop_resource: B
4sequential_4_dense_9_biasadd_readvariableop_resource:
identity¢<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp¢>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1¢>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2¢@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp¢<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp¢>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1¢>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2¢@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp¢<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp¢>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1¢>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2¢@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp¢+sequential_4/dense_8/BiasAdd/ReadVariableOp¢*sequential_4/dense_8/MatMul/ReadVariableOp¢+sequential_4/dense_9/BiasAdd/ReadVariableOp¢*sequential_4/dense_9/MatMul/ReadVariableOp¢8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp¢7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp¢9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp¢sequential_4/lstm_12/while¢8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢sequential_4/lstm_13/while¢8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp¢7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp¢9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp¢sequential_4/lstm_14/whileW
sequential_4/lstm_12/ShapeShapelstm_12_input*
T0*
_output_shapes
:r
(sequential_4/lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_4/lstm_12/strided_sliceStridedSlice#sequential_4/lstm_12/Shape:output:01sequential_4/lstm_12/strided_slice/stack:output:03sequential_4/lstm_12/strided_slice/stack_1:output:03sequential_4/lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#sequential_4/lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :²
!sequential_4/lstm_12/zeros/packedPack+sequential_4/lstm_12/strided_slice:output:0,sequential_4/lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_4/lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
sequential_4/lstm_12/zerosFill*sequential_4/lstm_12/zeros/packed:output:0)sequential_4/lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
%sequential_4/lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¶
#sequential_4/lstm_12/zeros_1/packedPack+sequential_4/lstm_12/strided_slice:output:0.sequential_4/lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_4/lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
sequential_4/lstm_12/zeros_1Fill,sequential_4/lstm_12/zeros_1/packed:output:0+sequential_4/lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#sequential_4/lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_4/lstm_12/transpose	Transposelstm_12_input,sequential_4/lstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:<ÿÿÿÿÿÿÿÿÿn
sequential_4/lstm_12/Shape_1Shape"sequential_4/lstm_12/transpose:y:0*
T0*
_output_shapes
:t
*sequential_4/lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$sequential_4/lstm_12/strided_slice_1StridedSlice%sequential_4/lstm_12/Shape_1:output:03sequential_4/lstm_12/strided_slice_1/stack:output:05sequential_4/lstm_12/strided_slice_1/stack_1:output:05sequential_4/lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_4/lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"sequential_4/lstm_12/TensorArrayV2TensorListReserve9sequential_4/lstm_12/TensorArrayV2/element_shape:output:0-sequential_4/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jsequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
<sequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_12/transpose:y:0Ssequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*sequential_4/lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$sequential_4/lstm_12/strided_slice_2StridedSlice"sequential_4/lstm_12/transpose:y:03sequential_4/lstm_12/strided_slice_2/stack:output:05sequential_4/lstm_12/strided_slice_2/stack_1:output:05sequential_4/lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¹
7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp@sequential_4_lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Õ
(sequential_4/lstm_12/lstm_cell_12/MatMulMatMul-sequential_4/lstm_12/strided_slice_2:output:0?sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpBsequential_4_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Ï
*sequential_4/lstm_12/lstm_cell_12/MatMul_1MatMul#sequential_4/lstm_12/zeros:output:0Asequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%sequential_4/lstm_12/lstm_cell_12/addAddV22sequential_4/lstm_12/lstm_cell_12/MatMul:product:04sequential_4/lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
)sequential_4/lstm_12/lstm_cell_12/BiasAddBiasAdd)sequential_4/lstm_12/lstm_cell_12/add:z:0@sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1sequential_4/lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential_4/lstm_12/lstm_cell_12/splitSplit:sequential_4/lstm_12/lstm_cell_12/split/split_dim:output:02sequential_4/lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)sequential_4/lstm_12/lstm_cell_12/SigmoidSigmoid0sequential_4/lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/lstm_12/lstm_cell_12/Sigmoid_1Sigmoid0sequential_4/lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
%sequential_4/lstm_12/lstm_cell_12/mulMul/sequential_4/lstm_12/lstm_cell_12/Sigmoid_1:y:0%sequential_4/lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_4/lstm_12/lstm_cell_12/ReluRelu0sequential_4/lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'sequential_4/lstm_12/lstm_cell_12/mul_1Mul-sequential_4/lstm_12/lstm_cell_12/Sigmoid:y:04sequential_4/lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'sequential_4/lstm_12/lstm_cell_12/add_1AddV2)sequential_4/lstm_12/lstm_cell_12/mul:z:0+sequential_4/lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/lstm_12/lstm_cell_12/Sigmoid_2Sigmoid0sequential_4/lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_4/lstm_12/lstm_cell_12/Relu_1Relu+sequential_4/lstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
'sequential_4/lstm_12/lstm_cell_12/mul_2Mul/sequential_4/lstm_12/lstm_cell_12/Sigmoid_2:y:06sequential_4/lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2sequential_4/lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ÷
$sequential_4/lstm_12/TensorArrayV2_1TensorListReserve;sequential_4/lstm_12/TensorArrayV2_1/element_shape:output:0-sequential_4/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
sequential_4/lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_4/lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'sequential_4/lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¬
sequential_4/lstm_12/whileWhile0sequential_4/lstm_12/while/loop_counter:output:06sequential_4/lstm_12/while/maximum_iterations:output:0"sequential_4/lstm_12/time:output:0-sequential_4/lstm_12/TensorArrayV2_1:handle:0#sequential_4/lstm_12/zeros:output:0%sequential_4/lstm_12/zeros_1:output:0-sequential_4/lstm_12/strided_slice_1:output:0Lsequential_4/lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_4_lstm_12_lstm_cell_12_matmul_readvariableop_resourceBsequential_4_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceAsequential_4_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_4_lstm_12_while_body_162140*2
cond*R(
&sequential_4_lstm_12_while_cond_162139*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Esequential_4/lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
7sequential_4/lstm_12/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_12/while:output:3Nsequential_4/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0}
*sequential_4/lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,sequential_4/lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
$sequential_4/lstm_12/strided_slice_3StridedSlice@sequential_4/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_12/strided_slice_3/stack:output:05sequential_4/lstm_12/strided_slice_3/stack_1:output:05sequential_4/lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskz
%sequential_4/lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ö
 sequential_4/lstm_12/transpose_1	Transpose@sequential_4/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<p
sequential_4/lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
 sequential_4/dropout_16/IdentityIdentity$sequential_4/lstm_12/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<¿
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0x
3sequential_4/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ä
1sequential_4/batch_normalization_12/batchnorm/addAddV2Dsequential_4/batch_normalization_12/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
3sequential_4/batch_normalization_12/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
:Ç
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0á
1sequential_4/batch_normalization_12/batchnorm/mulMul7sequential_4/batch_normalization_12/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ó
3sequential_4/batch_normalization_12/batchnorm/mul_1Mul)sequential_4/dropout_16/Identity:output:05sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<Ã
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
3sequential_4/batch_normalization_12/batchnorm/mul_2MulFsequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ã
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ß
1sequential_4/batch_normalization_12/batchnorm/subSubFsequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ã
3sequential_4/batch_normalization_12/batchnorm/add_1AddV27sequential_4/batch_normalization_12/batchnorm/mul_1:z:05sequential_4/batch_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
sequential_4/lstm_13/ShapeShape7sequential_4/batch_normalization_12/batchnorm/add_1:z:0*
T0*
_output_shapes
:r
(sequential_4/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_4/lstm_13/strided_sliceStridedSlice#sequential_4/lstm_13/Shape:output:01sequential_4/lstm_13/strided_slice/stack:output:03sequential_4/lstm_13/strided_slice/stack_1:output:03sequential_4/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#sequential_4/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :²
!sequential_4/lstm_13/zeros/packedPack+sequential_4/lstm_13/strided_slice:output:0,sequential_4/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_4/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
sequential_4/lstm_13/zerosFill*sequential_4/lstm_13/zeros/packed:output:0)sequential_4/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
%sequential_4/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¶
#sequential_4/lstm_13/zeros_1/packedPack+sequential_4/lstm_13/strided_slice:output:0.sequential_4/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_4/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
sequential_4/lstm_13/zeros_1Fill,sequential_4/lstm_13/zeros_1/packed:output:0+sequential_4/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#sequential_4/lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
sequential_4/lstm_13/transpose	Transpose7sequential_4/batch_normalization_12/batchnorm/add_1:z:0,sequential_4/lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿn
sequential_4/lstm_13/Shape_1Shape"sequential_4/lstm_13/transpose:y:0*
T0*
_output_shapes
:t
*sequential_4/lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$sequential_4/lstm_13/strided_slice_1StridedSlice%sequential_4/lstm_13/Shape_1:output:03sequential_4/lstm_13/strided_slice_1/stack:output:05sequential_4/lstm_13/strided_slice_1/stack_1:output:05sequential_4/lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_4/lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"sequential_4/lstm_13/TensorArrayV2TensorListReserve9sequential_4/lstm_13/TensorArrayV2/element_shape:output:0-sequential_4/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jsequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
<sequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_13/transpose:y:0Ssequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*sequential_4/lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
$sequential_4/lstm_13/strided_slice_2StridedSlice"sequential_4/lstm_13/transpose:y:03sequential_4/lstm_13/strided_slice_2/stack:output:05sequential_4/lstm_13/strided_slice_2/stack_1:output:05sequential_4/lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskº
7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp@sequential_4_lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Õ
(sequential_4/lstm_13/lstm_cell_13/MatMulMatMul-sequential_4/lstm_13/strided_slice_2:output:0?sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpBsequential_4_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Ï
*sequential_4/lstm_13/lstm_cell_13/MatMul_1MatMul#sequential_4/lstm_13/zeros:output:0Asequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%sequential_4/lstm_13/lstm_cell_13/addAddV22sequential_4/lstm_13/lstm_cell_13/MatMul:product:04sequential_4/lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
)sequential_4/lstm_13/lstm_cell_13/BiasAddBiasAdd)sequential_4/lstm_13/lstm_cell_13/add:z:0@sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1sequential_4/lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential_4/lstm_13/lstm_cell_13/splitSplit:sequential_4/lstm_13/lstm_cell_13/split/split_dim:output:02sequential_4/lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)sequential_4/lstm_13/lstm_cell_13/SigmoidSigmoid0sequential_4/lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/lstm_13/lstm_cell_13/Sigmoid_1Sigmoid0sequential_4/lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
%sequential_4/lstm_13/lstm_cell_13/mulMul/sequential_4/lstm_13/lstm_cell_13/Sigmoid_1:y:0%sequential_4/lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_4/lstm_13/lstm_cell_13/ReluRelu0sequential_4/lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'sequential_4/lstm_13/lstm_cell_13/mul_1Mul-sequential_4/lstm_13/lstm_cell_13/Sigmoid:y:04sequential_4/lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'sequential_4/lstm_13/lstm_cell_13/add_1AddV2)sequential_4/lstm_13/lstm_cell_13/mul:z:0+sequential_4/lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/lstm_13/lstm_cell_13/Sigmoid_2Sigmoid0sequential_4/lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_4/lstm_13/lstm_cell_13/Relu_1Relu+sequential_4/lstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
'sequential_4/lstm_13/lstm_cell_13/mul_2Mul/sequential_4/lstm_13/lstm_cell_13/Sigmoid_2:y:06sequential_4/lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2sequential_4/lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ÷
$sequential_4/lstm_13/TensorArrayV2_1TensorListReserve;sequential_4/lstm_13/TensorArrayV2_1/element_shape:output:0-sequential_4/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
sequential_4/lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_4/lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'sequential_4/lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¬
sequential_4/lstm_13/whileWhile0sequential_4/lstm_13/while/loop_counter:output:06sequential_4/lstm_13/while/maximum_iterations:output:0"sequential_4/lstm_13/time:output:0-sequential_4/lstm_13/TensorArrayV2_1:handle:0#sequential_4/lstm_13/zeros:output:0%sequential_4/lstm_13/zeros_1:output:0-sequential_4/lstm_13/strided_slice_1:output:0Lsequential_4/lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_4_lstm_13_lstm_cell_13_matmul_readvariableop_resourceBsequential_4_lstm_13_lstm_cell_13_matmul_1_readvariableop_resourceAsequential_4_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_4_lstm_13_while_body_162296*2
cond*R(
&sequential_4_lstm_13_while_cond_162295*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Esequential_4/lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
7sequential_4/lstm_13/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_13/while:output:3Nsequential_4/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0}
*sequential_4/lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,sequential_4/lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
$sequential_4/lstm_13/strided_slice_3StridedSlice@sequential_4/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_13/strided_slice_3/stack:output:05sequential_4/lstm_13/strided_slice_3/stack_1:output:05sequential_4/lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskz
%sequential_4/lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ö
 sequential_4/lstm_13/transpose_1	Transpose@sequential_4/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<p
sequential_4/lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
 sequential_4/dropout_17/IdentityIdentity$sequential_4/lstm_13/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<¿
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0x
3sequential_4/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ä
1sequential_4/batch_normalization_13/batchnorm/addAddV2Dsequential_4/batch_normalization_13/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
3sequential_4/batch_normalization_13/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes	
:Ç
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0á
1sequential_4/batch_normalization_13/batchnorm/mulMul7sequential_4/batch_normalization_13/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ó
3sequential_4/batch_normalization_13/batchnorm/mul_1Mul)sequential_4/dropout_17/Identity:output:05sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<Ã
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
3sequential_4/batch_normalization_13/batchnorm/mul_2MulFsequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ã
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ß
1sequential_4/batch_normalization_13/batchnorm/subSubFsequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ã
3sequential_4/batch_normalization_13/batchnorm/add_1AddV27sequential_4/batch_normalization_13/batchnorm/mul_1:z:05sequential_4/batch_normalization_13/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
sequential_4/lstm_14/ShapeShape7sequential_4/batch_normalization_13/batchnorm/add_1:z:0*
T0*
_output_shapes
:r
(sequential_4/lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_4/lstm_14/strided_sliceStridedSlice#sequential_4/lstm_14/Shape:output:01sequential_4/lstm_14/strided_slice/stack:output:03sequential_4/lstm_14/strided_slice/stack_1:output:03sequential_4/lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#sequential_4/lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :²
!sequential_4/lstm_14/zeros/packedPack+sequential_4/lstm_14/strided_slice:output:0,sequential_4/lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_4/lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¬
sequential_4/lstm_14/zerosFill*sequential_4/lstm_14/zeros/packed:output:0)sequential_4/lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
%sequential_4/lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¶
#sequential_4/lstm_14/zeros_1/packedPack+sequential_4/lstm_14/strided_slice:output:0.sequential_4/lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_4/lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
sequential_4/lstm_14/zeros_1Fill,sequential_4/lstm_14/zeros_1/packed:output:0+sequential_4/lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#sequential_4/lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
sequential_4/lstm_14/transpose	Transpose7sequential_4/batch_normalization_13/batchnorm/add_1:z:0,sequential_4/lstm_14/transpose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿn
sequential_4/lstm_14/Shape_1Shape"sequential_4/lstm_14/transpose:y:0*
T0*
_output_shapes
:t
*sequential_4/lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$sequential_4/lstm_14/strided_slice_1StridedSlice%sequential_4/lstm_14/Shape_1:output:03sequential_4/lstm_14/strided_slice_1/stack:output:05sequential_4/lstm_14/strided_slice_1/stack_1:output:05sequential_4/lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_4/lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"sequential_4/lstm_14/TensorArrayV2TensorListReserve9sequential_4/lstm_14/TensorArrayV2/element_shape:output:0-sequential_4/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jsequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
<sequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_14/transpose:y:0Ssequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*sequential_4/lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
$sequential_4/lstm_14/strided_slice_2StridedSlice"sequential_4/lstm_14/transpose:y:03sequential_4/lstm_14/strided_slice_2/stack:output:05sequential_4/lstm_14/strided_slice_2/stack_1:output:05sequential_4/lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskº
7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp@sequential_4_lstm_14_lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Õ
(sequential_4/lstm_14/lstm_cell_14/MatMulMatMul-sequential_4/lstm_14/strided_slice_2:output:0?sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpBsequential_4_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Ï
*sequential_4/lstm_14/lstm_cell_14/MatMul_1MatMul#sequential_4/lstm_14/zeros:output:0Asequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%sequential_4/lstm_14/lstm_cell_14/addAddV22sequential_4/lstm_14/lstm_cell_14/MatMul:product:04sequential_4/lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
)sequential_4/lstm_14/lstm_cell_14/BiasAddBiasAdd)sequential_4/lstm_14/lstm_cell_14/add:z:0@sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1sequential_4/lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential_4/lstm_14/lstm_cell_14/splitSplit:sequential_4/lstm_14/lstm_cell_14/split/split_dim:output:02sequential_4/lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)sequential_4/lstm_14/lstm_cell_14/SigmoidSigmoid0sequential_4/lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/lstm_14/lstm_cell_14/Sigmoid_1Sigmoid0sequential_4/lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
%sequential_4/lstm_14/lstm_cell_14/mulMul/sequential_4/lstm_14/lstm_cell_14/Sigmoid_1:y:0%sequential_4/lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_4/lstm_14/lstm_cell_14/ReluRelu0sequential_4/lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'sequential_4/lstm_14/lstm_cell_14/mul_1Mul-sequential_4/lstm_14/lstm_cell_14/Sigmoid:y:04sequential_4/lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'sequential_4/lstm_14/lstm_cell_14/add_1AddV2)sequential_4/lstm_14/lstm_cell_14/mul:z:0+sequential_4/lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/lstm_14/lstm_cell_14/Sigmoid_2Sigmoid0sequential_4/lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_4/lstm_14/lstm_cell_14/Relu_1Relu+sequential_4/lstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
'sequential_4/lstm_14/lstm_cell_14/mul_2Mul/sequential_4/lstm_14/lstm_cell_14/Sigmoid_2:y:06sequential_4/lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2sequential_4/lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ÷
$sequential_4/lstm_14/TensorArrayV2_1TensorListReserve;sequential_4/lstm_14/TensorArrayV2_1/element_shape:output:0-sequential_4/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
sequential_4/lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_4/lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'sequential_4/lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¬
sequential_4/lstm_14/whileWhile0sequential_4/lstm_14/while/loop_counter:output:06sequential_4/lstm_14/while/maximum_iterations:output:0"sequential_4/lstm_14/time:output:0-sequential_4/lstm_14/TensorArrayV2_1:handle:0#sequential_4/lstm_14/zeros:output:0%sequential_4/lstm_14/zeros_1:output:0-sequential_4/lstm_14/strided_slice_1:output:0Lsequential_4/lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_4_lstm_14_lstm_cell_14_matmul_readvariableop_resourceBsequential_4_lstm_14_lstm_cell_14_matmul_1_readvariableop_resourceAsequential_4_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_4_lstm_14_while_body_162452*2
cond*R(
&sequential_4_lstm_14_while_cond_162451*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Esequential_4/lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
7sequential_4/lstm_14/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_14/while:output:3Nsequential_4/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0}
*sequential_4/lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,sequential_4/lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
$sequential_4/lstm_14/strided_slice_3StridedSlice@sequential_4/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_14/strided_slice_3/stack:output:05sequential_4/lstm_14/strided_slice_3/stack_1:output:05sequential_4/lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskz
%sequential_4/lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ö
 sequential_4/lstm_14/transpose_1	Transpose@sequential_4/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<p
sequential_4/lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
 sequential_4/dropout_18/IdentityIdentity-sequential_4/lstm_14/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0x
3sequential_4/batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ä
1sequential_4/batch_normalization_14/batchnorm/addAddV2Dsequential_4/batch_normalization_14/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
3sequential_4/batch_normalization_14/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes	
:Ç
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0á
1sequential_4/batch_normalization_14/batchnorm/mulMul7sequential_4/batch_normalization_14/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ï
3sequential_4/batch_normalization_14/batchnorm/mul_1Mul)sequential_4/dropout_18/Identity:output:05sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_14_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
3sequential_4/batch_normalization_14/batchnorm/mul_2MulFsequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ã
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_14_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ß
1sequential_4/batch_normalization_14/batchnorm/subSubFsequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ß
3sequential_4/batch_normalization_14/batchnorm/add_1AddV27sequential_4/batch_normalization_14/batchnorm/mul_1:z:05sequential_4/batch_normalization_14/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_8_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0Ä
sequential_4/dense_8/MatMulMatMul7sequential_4/batch_normalization_14/batchnorm/add_1:z:02sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_4/dense_8/BiasAddBiasAdd%sequential_4/dense_8/MatMul:product:03sequential_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
sequential_4/dense_8/ReluRelu%sequential_4/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 sequential_4/dropout_19/IdentityIdentity'sequential_4/dense_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential_4/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0¶
sequential_4/dense_9/MatMulMatMul)sequential_4/dropout_19/Identity:output:02sequential_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_4/dense_9/BiasAddBiasAdd%sequential_4/dense_9/MatMul:product:03sequential_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/dense_9/SoftmaxSoftmax%sequential_4/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&sequential_4/dense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp=^sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp,^sequential_4/dense_8/BiasAdd/ReadVariableOp+^sequential_4/dense_8/MatMul/ReadVariableOp,^sequential_4/dense_9/BiasAdd/ReadVariableOp+^sequential_4/dense_9/MatMul/ReadVariableOp9^sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8^sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:^sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^sequential_4/lstm_12/while9^sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8^sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:^sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^sequential_4/lstm_13/while9^sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp8^sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:^sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^sequential_4/lstm_14/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ<: : : : : : : : : : : : : : : : : : : : : : : : : 2|
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp2
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_12
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_22
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp2|
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp2
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_12
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_22
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp2|
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp2
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_12
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_22
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp2Z
+sequential_4/dense_8/BiasAdd/ReadVariableOp+sequential_4/dense_8/BiasAdd/ReadVariableOp2X
*sequential_4/dense_8/MatMul/ReadVariableOp*sequential_4/dense_8/MatMul/ReadVariableOp2Z
+sequential_4/dense_9/BiasAdd/ReadVariableOp+sequential_4/dense_9/BiasAdd/ReadVariableOp2X
*sequential_4/dense_9/MatMul/ReadVariableOp*sequential_4/dense_9/MatMul/ReadVariableOp2t
8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8sequential_4/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2r
7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp7sequential_4/lstm_12/lstm_cell_12/MatMul/ReadVariableOp2v
9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp9sequential_4/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp28
sequential_4/lstm_12/whilesequential_4/lstm_12/while2t
8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8sequential_4/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2r
7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp7sequential_4/lstm_13/lstm_cell_13/MatMul/ReadVariableOp2v
9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp9sequential_4/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp28
sequential_4/lstm_13/whilesequential_4/lstm_13/while2t
8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp8sequential_4/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2r
7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp7sequential_4/lstm_14/lstm_cell_14/MatMul/ReadVariableOp2v
9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp9sequential_4/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp28
sequential_4/lstm_14/whilesequential_4/lstm_14/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
'
_user_specified_namelstm_12_input


õ
C__inference_dense_8_layer_call_and_return_conditional_losses_164380

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_162635

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¹
Ã
while_cond_163080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163080___redundant_placeholder04
0while_while_cond_163080___redundant_placeholder14
0while_while_cond_163080___redundant_placeholder24
0while_while_cond_163080___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


ã
lstm_13_while_cond_166272,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1D
@lstm_13_while_lstm_13_while_cond_166272___redundant_placeholder0D
@lstm_13_while_lstm_13_while_cond_166272___redundant_placeholder1D
@lstm_13_while_lstm_13_while_cond_166272___redundant_placeholder2D
@lstm_13_while_lstm_13_while_cond_166272___redundant_placeholder3
lstm_13_while_identity

lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: [
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_13_while_identitylstm_13/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
£
µ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_167340

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£8

C__inference_lstm_14_layer_call_and_return_conditional_losses_163773

inputs'
lstm_cell_14_163691:
'
lstm_cell_14_163693:
"
lstm_cell_14_163695:	
identity¢$lstm_cell_14/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskø
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_163691lstm_cell_14_163693lstm_cell_14_163695*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163645n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_163691lstm_cell_14_163693lstm_cell_14_163695*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163704*
condR
while_cond_163703*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
òA
Ô

lstm_14_while_body_166450,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:
Q
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
K
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorM
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:
O
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
I
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	¢1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp¢0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp¢2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0®
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ò
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¹
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¿
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: q
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: ±
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_14/while/NoOpNoOp2^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"È
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
þJ

C__inference_lstm_14_layer_call_and_return_conditional_losses_168284
inputs_0?
+lstm_cell_14_matmul_readvariableop_resource:
A
-lstm_cell_14_matmul_1_readvariableop_resource:
;
,lstm_cell_14_biasadd_readvariableop_resource:	
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_168200*
condR
while_cond_168199*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
üJ

C__inference_lstm_12_layer_call_and_return_conditional_losses_166981
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	A
-lstm_cell_12_matmul_1_readvariableop_resource:
;
,lstm_cell_12_biasadd_readvariableop_resource:	
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_166897*
condR
while_cond_166896*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
£
µ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_168063

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£8

C__inference_lstm_14_layer_call_and_return_conditional_losses_163582

inputs'
lstm_cell_14_163500:
'
lstm_cell_14_163502:
"
lstm_cell_14_163504:	
identity¢$lstm_cell_14/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskø
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_163500lstm_cell_14_163502lstm_cell_14_163504*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_163499n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_163500lstm_cell_14_163502lstm_cell_14_163504*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163513*
condR
while_cond_163512*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ã
while_cond_163928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163928___redundant_placeholder04
0while_while_cond_163928___redundant_placeholder14
0while_while_cond_163928___redundant_placeholder24
0while_while_cond_163928___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


e
F__inference_dropout_17_layer_call_and_return_conditional_losses_164715

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹
Ã
while_cond_164601
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_164601___redundant_placeholder04
0while_while_cond_164601___redundant_placeholder14
0while_while_cond_164601___redundant_placeholder24
0while_while_cond_164601___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹8
Ô
while_body_167763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
A
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

ç
&sequential_4_lstm_14_while_cond_162451F
Bsequential_4_lstm_14_while_sequential_4_lstm_14_while_loop_counterL
Hsequential_4_lstm_14_while_sequential_4_lstm_14_while_maximum_iterations*
&sequential_4_lstm_14_while_placeholder,
(sequential_4_lstm_14_while_placeholder_1,
(sequential_4_lstm_14_while_placeholder_2,
(sequential_4_lstm_14_while_placeholder_3H
Dsequential_4_lstm_14_while_less_sequential_4_lstm_14_strided_slice_1^
Zsequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_162451___redundant_placeholder0^
Zsequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_162451___redundant_placeholder1^
Zsequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_162451___redundant_placeholder2^
Zsequential_4_lstm_14_while_sequential_4_lstm_14_while_cond_162451___redundant_placeholder3'
#sequential_4_lstm_14_while_identity
¶
sequential_4/lstm_14/while/LessLess&sequential_4_lstm_14_while_placeholderDsequential_4_lstm_14_while_less_sequential_4_lstm_14_strided_slice_1*
T0*
_output_shapes
: u
#sequential_4/lstm_14/while/IdentityIdentity#sequential_4/lstm_14/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_4_lstm_14_while_identity,sequential_4/lstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹
Ã
while_cond_168199
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_168199___redundant_placeholder04
0while_while_cond_168199___redundant_placeholder14
0while_while_cond_168199___redundant_placeholder24
0while_while_cond_168199___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
K

C__inference_lstm_13_layer_call_and_return_conditional_losses_167561
inputs_0?
+lstm_cell_13_matmul_readvariableop_resource:
A
-lstm_cell_13_matmul_1_readvariableop_resource:
;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_167477*
condR
while_cond_167476*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
í
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_167282

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¾J

C__inference_lstm_12_layer_call_and_return_conditional_losses_167267

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	A
-lstm_cell_12_matmul_1_readvariableop_resource:
;
,lstm_cell_12_biasadd_readvariableop_resource:	
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_167183*
condR
while_cond_167182*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
ÄJ

C__inference_lstm_13_layer_call_and_return_conditional_losses_167847

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
A
-lstm_cell_13_matmul_1_readvariableop_resource:
;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_167763*
condR
while_cond_167762*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<À
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_168728

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_169051

inputs
states_0
states_12
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
î
#
__inference__traced_save_169420
file_prefix;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableopD
@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop8
4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop:
6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableopD
@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop8
4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop:
6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableopD
@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop8
4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_14_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_14_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_14_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_14_bias_v_read_readvariableop
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
: %
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*¼$
value²$B¯$IB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*§
valueBIB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B "
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableop@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableop@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_batch_normalization_12_gamma_m_read_readvariableop=savev2_adam_batch_normalization_12_beta_m_read_readvariableop>savev2_adam_batch_normalization_13_gamma_m_read_readvariableop=savev2_adam_batch_normalization_13_beta_m_read_readvariableop>savev2_adam_batch_normalization_14_gamma_m_read_readvariableop=savev2_adam_batch_normalization_14_beta_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableop=savev2_adam_lstm_14_lstm_cell_14_kernel_m_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_14_lstm_cell_14_bias_m_read_readvariableop>savev2_adam_batch_normalization_12_gamma_v_read_readvariableop=savev2_adam_batch_normalization_12_beta_v_read_readvariableop>savev2_adam_batch_normalization_13_gamma_v_read_readvariableop=savev2_adam_batch_normalization_13_beta_v_read_readvariableop>savev2_adam_batch_normalization_14_gamma_v_read_readvariableop=savev2_adam_batch_normalization_14_beta_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableop=savev2_adam_lstm_14_lstm_cell_14_kernel_v_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_14_lstm_cell_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *W
dtypesM
K2I	
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

identity_1Identity_1:output:0*Ê
_input_shapes¸
µ: :::::::::::::	 : : :: : : : : :	:
::
:
::
:
:: : : : :::::::	 : : ::	:
::
:
::
:
::::::::	 : : ::	:
::
:
::
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	 : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :!#

_output_shapes	
::!$

_output_shapes	
::!%

_output_shapes	
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::%)!

_output_shapes
:	 : *

_output_shapes
: :$+ 

_output_shapes

: : ,

_output_shapes
::%-!

_output_shapes
:	:&."
 
_output_shapes
:
:!/

_output_shapes	
::&0"
 
_output_shapes
:
:&1"
 
_output_shapes
:
:!2

_output_shapes	
::&3"
 
_output_shapes
:
:&4"
 
_output_shapes
:
:!5

_output_shapes	
::!6

_output_shapes	
::!7

_output_shapes	
::!8

_output_shapes	
::!9

_output_shapes	
::!:

_output_shapes	
::!;

_output_shapes	
::%<!

_output_shapes
:	 : =

_output_shapes
: :$> 

_output_shapes

: : ?

_output_shapes
::%@!

_output_shapes
:	:&A"
 
_output_shapes
:
:!B

_output_shapes	
::&C"
 
_output_shapes
:
:&D"
 
_output_shapes
:
:!E

_output_shapes	
::&F"
 
_output_shapes
:
:&G"
 
_output_shapes
:
:!H

_output_shapes	
::I

_output_shapes
: 


õ
C__inference_dense_8_layer_call_and_return_conditional_losses_168840

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


e
F__inference_dropout_17_layer_call_and_return_conditional_losses_168017

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
£
µ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162942

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
Ö
7__inference_batch_normalization_12_layer_call_fn_167320

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_162989}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
&
ï
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_167374

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       £
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ã
lstm_14_while_cond_165920,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1D
@lstm_14_while_lstm_14_while_cond_165920___redundant_placeholder0D
@lstm_14_while_lstm_14_while_cond_165920___redundant_placeholder1D
@lstm_14_while_lstm_14_while_cond_165920___redundant_placeholder2D
@lstm_14_while_lstm_14_while_cond_165920___redundant_placeholder3
lstm_14_while_identity

lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: [
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_14_while_identitylstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¹8
Ô
while_body_167906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
A
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0º
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¡
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0§
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹
Ã
while_cond_166753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_166753___redundant_placeholder04
0while_while_cond_166753___redundant_placeholder14
0while_while_cond_166753___redundant_placeholder24
0while_while_cond_166753___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÉJ

C__inference_lstm_14_layer_call_and_return_conditional_losses_168713

inputs?
+lstm_cell_14_matmul_readvariableop_resource:
A
-lstm_cell_14_matmul_1_readvariableop_resource:
;
,lstm_cell_14_biasadd_readvariableop_resource:	
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_168629*
condR
while_cond_168628*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:<ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ<: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*º
serving_default¦
K
lstm_12_input:
serving_default_lstm_12_input:0ÿÿÿÿÿÿÿÿÿ<;
dense_90
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:þæ
Ô
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
	variables
 trainable_variables
!regularization_losses
"	keras_api
#_random_generator
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
1cell
2
state_spec
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
Lcell
M
state_spec
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R_random_generator
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
»

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
»

vkernel
wbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
~iter

beta_1
beta_2

decay
learning_rate'm(mBmCm]m^mgmhmvmwm	m	m	m	m	m	m	m	m	m'v(vBvCv]v^vgvhvvvwv	v 	v¡	v¢	v£	v¤	v¥	v¦	v§	v¨"
	optimizer
ç
0
1
2
'3
(4
)5
*6
7
8
9
B10
C11
D12
E13
14
15
16
]17
^18
_19
`20
g21
h22
v23
w24"
trackable_list_wrapper
·
0
1
2
'3
(4
5
6
7
B8
C9
10
11
12
]13
^14
g15
h16
v17
w18"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2ÿ
-__inference_sequential_4_layer_call_fn_164464
-__inference_sequential_4_layer_call_fn_165495
-__inference_sequential_4_layer_call_fn_165550
-__inference_sequential_4_layer_call_fn_165302À
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
î2ë
H__inference_sequential_4_layer_call_and_return_conditional_losses_166037
H__inference_sequential_4_layer_call_and_return_conditional_losses_166594
H__inference_sequential_4_layer_call_and_return_conditional_losses_165368
H__inference_sequential_4_layer_call_and_return_conditional_losses_165434À
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
ÒBÏ
!__inference__wrapped_model_162568lstm_12_input"
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
-
serving_default"
signature_map


state_size
kernel
recurrent_kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¿
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_lstm_12_layer_call_fn_166662
(__inference_lstm_12_layer_call_fn_166673
(__inference_lstm_12_layer_call_fn_166684
(__inference_lstm_12_layer_call_fn_166695Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_12_layer_call_and_return_conditional_losses_166838
C__inference_lstm_12_layer_call_and_return_conditional_losses_166981
C__inference_lstm_12_layer_call_and_return_conditional_losses_167124
C__inference_lstm_12_layer_call_and_return_conditional_losses_167267Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
 trainable_variables
!regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_16_layer_call_fn_167272
+__inference_dropout_16_layer_call_fn_167277´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_16_layer_call_and_return_conditional_losses_167282
F__inference_dropout_16_layer_call_and_return_conditional_losses_167294´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_12/gamma
*:(2batch_normalization_12/beta
3:1 (2"batch_normalization_12/moving_mean
7:5 (2&batch_normalization_12/moving_variance
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
¬2©
7__inference_batch_normalization_12_layer_call_fn_167307
7__inference_batch_normalization_12_layer_call_fn_167320´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_167340
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_167374´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 

ª
state_size
kernel
recurrent_kernel
	bias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯_random_generator
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¿
²states
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_lstm_13_layer_call_fn_167385
(__inference_lstm_13_layer_call_fn_167396
(__inference_lstm_13_layer_call_fn_167407
(__inference_lstm_13_layer_call_fn_167418Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_13_layer_call_and_return_conditional_losses_167561
C__inference_lstm_13_layer_call_and_return_conditional_losses_167704
C__inference_lstm_13_layer_call_and_return_conditional_losses_167847
C__inference_lstm_13_layer_call_and_return_conditional_losses_167990Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_17_layer_call_fn_167995
+__inference_dropout_17_layer_call_fn_168000´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_17_layer_call_and_return_conditional_losses_168005
F__inference_dropout_17_layer_call_and_return_conditional_losses_168017´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_13/gamma
*:(2batch_normalization_13/beta
3:1 (2"batch_normalization_13/moving_mean
7:5 (2&batch_normalization_13/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
¬2©
7__inference_batch_normalization_13_layer_call_fn_168030
7__inference_batch_normalization_13_layer_call_fn_168043´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_168063
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_168097´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 

Â
state_size
kernel
recurrent_kernel
	bias
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç_random_generator
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¿
Êstates
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_lstm_14_layer_call_fn_168108
(__inference_lstm_14_layer_call_fn_168119
(__inference_lstm_14_layer_call_fn_168130
(__inference_lstm_14_layer_call_fn_168141Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_14_layer_call_and_return_conditional_losses_168284
C__inference_lstm_14_layer_call_and_return_conditional_losses_168427
C__inference_lstm_14_layer_call_and_return_conditional_losses_168570
C__inference_lstm_14_layer_call_and_return_conditional_losses_168713Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_18_layer_call_fn_168718
+__inference_dropout_18_layer_call_fn_168723´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_18_layer_call_and_return_conditional_losses_168728
F__inference_dropout_18_layer_call_and_return_conditional_losses_168740´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
+:)2batch_normalization_14/gamma
*:(2batch_normalization_14/beta
3:1 (2"batch_normalization_14/moving_mean
7:5 (2&batch_normalization_14/moving_variance
<
]0
^1
_2
`3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
¬2©
7__inference_batch_normalization_14_layer_call_fn_168753
7__inference_batch_normalization_14_layer_call_fn_168766´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_168786
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_168820´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:	 2dense_8/kernel
: 2dense_8/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_8_layer_call_fn_168829¢
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
í2ê
C__inference_dense_8_layer_call_and_return_conditional_losses_168840¢
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
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_19_layer_call_fn_168845
+__inference_dropout_19_layer_call_fn_168850´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_19_layer_call_and_return_conditional_losses_168855
F__inference_dropout_19_layer_call_and_return_conditional_losses_168867´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 : 2dense_9/kernel
:2dense_9/bias
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_9_layer_call_fn_168876¢
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
í2ê
C__inference_dense_9_layer_call_and_return_conditional_losses_168887¢
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
.:,	2lstm_12/lstm_cell_12/kernel
9:7
2%lstm_12/lstm_cell_12/recurrent_kernel
(:&2lstm_12/lstm_cell_12/bias
/:-
2lstm_13/lstm_cell_13/kernel
9:7
2%lstm_13/lstm_cell_13/recurrent_kernel
(:&2lstm_13/lstm_cell_13/bias
/:-
2lstm_14/lstm_cell_14/kernel
9:7
2%lstm_14/lstm_cell_14/recurrent_kernel
(:&2lstm_14/lstm_cell_14/bias
J
)0
*1
D2
E3
_4
`5"
trackable_list_wrapper
v
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
10
11"
trackable_list_wrapper
0
é0
ê1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÑBÎ
$__inference_signature_wrapper_166651lstm_12_input"
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
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¢2
-__inference_lstm_cell_12_layer_call_fn_168904
-__inference_lstm_cell_12_layer_call_fn_168921¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_168953
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_168985¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
.
)0
*1"
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
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¢2
-__inference_lstm_cell_13_layer_call_fn_169002
-__inference_lstm_cell_13_layer_call_fn_169019¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_169051
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_169083¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
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
.
D0
E1"
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
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¢2
-__inference_lstm_cell_14_layer_call_fn_169100
-__inference_lstm_cell_14_layer_call_fn_169117¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_169149
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_169181¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
L0"
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
.
_0
`1"
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

útotal

ûcount
ü	variables
ý	keras_api"
_tf_keras_metric
c

þtotal

ÿcount

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
ú0
û1"
trackable_list_wrapper
.
ü	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
þ0
ÿ1"
trackable_list_wrapper
.
	variables"
_generic_user_object
0:.2#Adam/batch_normalization_12/gamma/m
/:-2"Adam/batch_normalization_12/beta/m
0:.2#Adam/batch_normalization_13/gamma/m
/:-2"Adam/batch_normalization_13/beta/m
0:.2#Adam/batch_normalization_14/gamma/m
/:-2"Adam/batch_normalization_14/beta/m
&:$	 2Adam/dense_8/kernel/m
: 2Adam/dense_8/bias/m
%:# 2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
3:1	2"Adam/lstm_12/lstm_cell_12/kernel/m
>:<
2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
-:+2 Adam/lstm_12/lstm_cell_12/bias/m
4:2
2"Adam/lstm_13/lstm_cell_13/kernel/m
>:<
2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
-:+2 Adam/lstm_13/lstm_cell_13/bias/m
4:2
2"Adam/lstm_14/lstm_cell_14/kernel/m
>:<
2,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m
-:+2 Adam/lstm_14/lstm_cell_14/bias/m
0:.2#Adam/batch_normalization_12/gamma/v
/:-2"Adam/batch_normalization_12/beta/v
0:.2#Adam/batch_normalization_13/gamma/v
/:-2"Adam/batch_normalization_13/beta/v
0:.2#Adam/batch_normalization_14/gamma/v
/:-2"Adam/batch_normalization_14/beta/v
&:$	 2Adam/dense_8/kernel/v
: 2Adam/dense_8/bias/v
%:# 2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
3:1	2"Adam/lstm_12/lstm_cell_12/kernel/v
>:<
2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
-:+2 Adam/lstm_12/lstm_cell_12/bias/v
4:2
2"Adam/lstm_13/lstm_cell_13/kernel/v
>:<
2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
-:+2 Adam/lstm_13/lstm_cell_13/bias/v
4:2
2"Adam/lstm_14/lstm_cell_14/kernel/v
>:<
2,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v
-:+2 Adam/lstm_14/lstm_cell_14/bias/v¹
!__inference__wrapped_model_162568"*')(EBDC`]_^ghvw:¢7
0¢-
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ<
ª "1ª.
,
dense_9!
dense_9ÿÿÿÿÿÿÿÿÿÔ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_167340~*')(A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_167374~)*'(A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
7__inference_batch_normalization_12_layer_call_fn_167307q*')(A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
7__inference_batch_normalization_12_layer_call_fn_167320q)*'(A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_168063~EBDCA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_168097~DEBCA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
7__inference_batch_normalization_13_layer_call_fn_168030qEBDCA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
7__inference_batch_normalization_13_layer_call_fn_168043qDEBCA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_168786d`]_^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 º
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_168820d_`]^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_14_layer_call_fn_168753W`]_^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_14_layer_call_fn_168766W_`]^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_8_layer_call_and_return_conditional_losses_168840]gh0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 |
(__inference_dense_8_layer_call_fn_168829Pgh0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ £
C__inference_dense_9_layer_call_and_return_conditional_losses_168887\vw/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_9_layer_call_fn_168876Ovw/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ°
F__inference_dropout_16_layer_call_and_return_conditional_losses_167282f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 °
F__inference_dropout_16_layer_call_and_return_conditional_losses_167294f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 
+__inference_dropout_16_layer_call_fn_167272Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª "ÿÿÿÿÿÿÿÿÿ<
+__inference_dropout_16_layer_call_fn_167277Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p
ª "ÿÿÿÿÿÿÿÿÿ<°
F__inference_dropout_17_layer_call_and_return_conditional_losses_168005f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 °
F__inference_dropout_17_layer_call_and_return_conditional_losses_168017f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 
+__inference_dropout_17_layer_call_fn_167995Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª "ÿÿÿÿÿÿÿÿÿ<
+__inference_dropout_17_layer_call_fn_168000Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ<
p
ª "ÿÿÿÿÿÿÿÿÿ<¨
F__inference_dropout_18_layer_call_and_return_conditional_losses_168728^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
F__inference_dropout_18_layer_call_and_return_conditional_losses_168740^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_18_layer_call_fn_168718Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_18_layer_call_fn_168723Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_19_layer_call_and_return_conditional_losses_168855\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¦
F__inference_dropout_19_layer_call_and_return_conditional_losses_168867\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_dropout_19_layer_call_fn_168845O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ ~
+__inference_dropout_19_layer_call_fn_168850O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ Ö
C__inference_lstm_12_layer_call_and_return_conditional_losses_166838O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ö
C__inference_lstm_12_layer_call_and_return_conditional_losses_166981O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¼
C__inference_lstm_12_layer_call_and_return_conditional_losses_167124u?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ<

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 ¼
C__inference_lstm_12_layer_call_and_return_conditional_losses_167267u?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ<

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 ®
(__inference_lstm_12_layer_call_fn_166662O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ®
(__inference_lstm_12_layer_call_fn_166673O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
(__inference_lstm_12_layer_call_fn_166684h?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ<

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ<
(__inference_lstm_12_layer_call_fn_166695h?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ<

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ<×
C__inference_lstm_13_layer_call_and_return_conditional_losses_167561P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ×
C__inference_lstm_13_layer_call_and_return_conditional_losses_167704P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ½
C__inference_lstm_13_layer_call_and_return_conditional_losses_167847v@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 ½
C__inference_lstm_13_layer_call_and_return_conditional_losses_167990v@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ<
 ¯
(__inference_lstm_13_layer_call_fn_167385P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
(__inference_lstm_13_layer_call_fn_167396P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
(__inference_lstm_13_layer_call_fn_167407i@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ<
(__inference_lstm_13_layer_call_fn_167418i@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ<Ê
C__inference_lstm_14_layer_call_and_return_conditional_losses_168284P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ê
C__inference_lstm_14_layer_call_and_return_conditional_losses_168427P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¹
C__inference_lstm_14_layer_call_and_return_conditional_losses_168570r@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¹
C__inference_lstm_14_layer_call_and_return_conditional_losses_168713r@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¡
(__inference_lstm_14_layer_call_fn_168108uP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
(__inference_lstm_14_layer_call_fn_168119uP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_14_layer_call_fn_168130e@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_14_layer_call_fn_168141e@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ<

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÒ
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_168953¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ò
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_168985¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 §
-__inference_lstm_cell_12_layer_call_fn_168904õ¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ§
-__inference_lstm_cell_12_layer_call_fn_168921õ¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÔ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_169051¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ô
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_169083¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ©
-__inference_lstm_cell_13_layer_call_fn_169002÷¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ©
-__inference_lstm_cell_13_layer_call_fn_169019÷¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÔ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_169149¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ô
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_169181¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ©
-__inference_lstm_cell_14_layer_call_fn_169100÷¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ©
-__inference_lstm_cell_14_layer_call_fn_169117÷¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÜ
H__inference_sequential_4_layer_call_and_return_conditional_losses_165368"*')(EBDC`]_^ghvwB¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ<
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
H__inference_sequential_4_layer_call_and_return_conditional_losses_165434")*'(DEBC_`]^ghvwB¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ<
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Õ
H__inference_sequential_4_layer_call_and_return_conditional_losses_166037"*')(EBDC`]_^ghvw;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ<
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Õ
H__inference_sequential_4_layer_call_and_return_conditional_losses_166594")*'(DEBC_`]^ghvw;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ<
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
-__inference_sequential_4_layer_call_fn_164464"*')(EBDC`]_^ghvwB¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ<
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
-__inference_sequential_4_layer_call_fn_165302")*'(DEBC_`]^ghvwB¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ<
p

 
ª "ÿÿÿÿÿÿÿÿÿ¬
-__inference_sequential_4_layer_call_fn_165495{"*')(EBDC`]_^ghvw;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ<
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¬
-__inference_sequential_4_layer_call_fn_165550{")*'(DEBC_`]^ghvw;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ<
p

 
ª "ÿÿÿÿÿÿÿÿÿÍ
$__inference_signature_wrapper_166651¤"*')(EBDC`]_^ghvwK¢H
¢ 
Aª>
<
lstm_12_input+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ<"1ª.
,
dense_9!
dense_9ÿÿÿÿÿÿÿÿÿ