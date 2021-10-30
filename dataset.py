
data = []
f = open("wdbc.data", "r")
# print(f.read())
data = list(f.read().split('\n'))
# # print(data)

# create array for store data
Dataset = []
for i in range(len(data)):
    str = list(data[i].split(','))
    Dataset.append(str)
# print(len(Dataset))

ID = []
Int_Input= []
Output_data = []

for i in Dataset:
    ID.append(i[0])
    Output_data.append(i[1])
    Input_data = (i[2:])
    Int_Input.append([float(i) for i in Input_data])
# print(Int_Input)

Desire = []
for i in range(len(Output_data)):
    if Output_data[i] == 'M':
        Desire.append('1')
    else:
        Desire.append('0')

def denomallize_list(x,ref=data):
    return [x[i] * (max(ref) - min(ref)) + min(ref) for i in range(len(x))]

def denomallize(x,ref=data):
    return x * (max(ref) - min(ref)) + min(ref)

def normalize_list(x,ref=data):
    return [((x[i]-min(ref))/(max(ref)-min(ref))) for i in range(len(x))]

def normalize(x,ref=data):
    return ((x-min(ref))/(max(ref)-min(ref)))

temp = [[Int_Input[j][i] for j in range(len(Int_Input))] for i in range(30)]
nomalized_input_colum_form = [normalize_list(temp[i],ref = temp[i]) for i in range(30)]
nomalized_input = [[nomalized_input_colum_form[j][i] for j in range(30)] for i in range(len(Int_Input))]

# print(nomalized_input_colum_form)
# print(nomalized_input)


