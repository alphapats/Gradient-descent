'''
------------------------------------------------------------------------------------------------------
File name: main.py
Created by: Maj Amit Pathania
Roll No: 163054001
Date:05 Mar 17
------------------------------------------------------------------------------------------------------
'''
import numpy as np

#function to read all colmuns of csv file
def read_fulldata(fname):
	data=np.genfromtxt(fname, skip_header=1, usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14),delimiter=',')
	data=data.reshape((len(data),14))
	return(data)

#function to read only attribute data from given csv file
def read_attr(fname):
	data=np.genfromtxt(fname, skip_header=1, usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13),delimiter=',')
	data=np.array(data)
	return(data)

#funtion to read OUTPUT(MEDV) values from csv file
def read_y(fname):
	data=np.genfromtxt(fname, skip_header=1, usecols=(14),delimiter=',')
	data=data.reshape((len(data),1))
	return(data)

#function to select specific column from data array
def get_column(data, colno):
    return [row[colno] for row in data]

'''
#split data into k folds
def cross_validation_split(X, K):
	#split a dataset into k folds
	split_set=list()
	dup_data=X
	fold_size=int(len(X) / K)
	for i in range(K):
		fold=list()
		idx=np.random.randint(dup_data.shape[0], size=fold_size)
		fold=dup_data[idx,:]
		split_set.append(fold)
	return(split_set)

def normalize(data,min_val,max_val,calc_mean):
	for row in data:
		for col in range(0,len(row)):
			if(min_val[col]<0):
				row[col]=row[col]+abs(min_val[col])
			else:
				row[col]=row[col]-abs(min_val[col])
			row[col] = (row[col]) / (max_val[col] - min_val[col])
	return(data)

'''
#function to normalise the data based on mean and standard deviation
def normalize(data1,calc_mean,dev):
	data=np.copy(data1)
	for row in data:
		for col in range(0,len(row)):
			row[col] = (row[col]-calc_mean[col]) / (dev[col])
	return(data)

#function to take log of attribute LSTAT col 12 and replacing that column
def log_lstat(attr):
    log_lstat=np.log(attr[:,[12]])
    attr=np.delete(attr,[12],1)
    attr=np.append(attr,log_lstat,axis=1)
    return(attr)
    
#function for feature engineering
def feature_man(attr):
    attr_copy=np.copy(attr) 
    for x in range(attr.shape[1]):
        for y in range(attr.shape[1]):
            temparr=[]
            for z in range(attr.shape[0]):
                temparr.append(attr[z][x]*attr[z][y]) #multiply each feature with every other feature
                
            attr_copy=np.insert(attr_copy,attr_copy.shape[1],temparr,axis=1)
    attr=attr_copy
    #taking square of features
    sq_features=np.square(attr[:,[0,1,2,3,4,5,6,8,11,12]])
    #taking cube of features
    cube_feature1=np.power(attr[:,[0,1,2,4,6,7,9,10,11,12]],3)
    #taking exponential of features
    exp_feature4=np.exp(attr[:,[12,2,5]])
    #appending square and cube terms into attribute matrix
    attr=np.concatenate((attr,sq_features,cube_feature1),axis=1)
    #appending exponential terms into attribute matrix
    attr=np.append(attr,exp_feature4,axis=1)
    #adding b term into attributes
    attr=np.insert(attr,0,np.ones((attr.shape[0])),axis=1)
    
    return(attr)

#removing outliners outside the range -7.5 and 7.5
def remove_outliners(attr):
	nrow,ncol=attr.shape
	outliners=[]
	for row in range(nrow):
		for col in range(ncol):
			if(attr[row][col]>7.5 or attr[row][col]<-7.5):
				outliners.append(row)
				break
	return(outliners)

#function to calculate w using closed form solution
def linear_reg(x,y):
    w=np.linalg.inv((x.T).dot(x)).dot(x.T).dot(y)
    return(w)

#function to calclate RMS error     
def costfn(phi,w,y):
	m=len(phi)
	err=np.subtract(np.matmul(phi,w),y)
	LSS=np.sum(err ** 2) / (m)  # cost
	RMS=np.power(LSS,0.5)
	return(err,RMS)

#function to calculate w using graient descent algorithm for linear regression
def gradientdescent(phi,y,step_size,conv_condn,max_iter,lamda):
	phitran=phi.transpose()
	m=len(y)
	#create intial w (ie w0)
	w=np.ones((phi.shape[1],1))
	num_iter=0
	cost_f = []
	converged=False
	while(num_iter<max_iter and converged ==False):
		num_iter=num_iter+1
		err,LSS=costfn(phi,w,y)
		cost_f.append(LSS)
		diff=1
		#take difference of error between k and k+1 step
		if(num_iter>2):
			prev=num_iter-2
			curr=num_iter-1
			diff=(cost_f[prev]-cost_f[curr])
		#check for convergence
		if(abs(diff)<conv_condn):
			converged=True
		else:
			grad=(1/m)*(np.dot(phitran,err)+ lamda*w)
			w=np.subtract(w,np.multiply(step_size,grad))
	return(w)

#fucntion to calculate w using gradient descent for lp norm
def gradientdescent_lp(phi,y,step_size,conv_condn,max_iter,lamda,p):
	phitran=phi.transpose()
	m=len(y)
	w=np.ones((phi.shape[1],1))
	num_iter=0
	cost_f = []
	converged=False
	
	while(num_iter<max_iter and converged ==False):
		num_iter=num_iter+1
		err,LSS=costfn(phi,w,y)
		cost_f.append(LSS)
		diff=1
		#take difference of error between k and k+1 step
		if(num_iter>2):
			prev=num_iter-2
			curr=num_iter-1
			diff=(cost_f[prev]-cost_f[curr])
			#print(diff)
		if(abs(diff)<conv_condn):
			converged=True
		else:
			grad=(1/m)*(np.dot(phitran,err))+ lamda*p*(np.power(np.absolute(w),p-1))
			w=np.subtract(w,np.multiply(step_size,grad))
	return(w)

#function to write the calculated y into output file
def writeoutput(testdatafile,phi,w,o_fname):
    y=np.matmul(phi,w)
    #read row id from test data
    row_id=np.genfromtxt(testdatafile, skip_header=1, usecols=(0),delimiter=',')
    result=np.column_stack((row_id,y))
    #open file and write to it
    np.savetxt(o_fname,result, fmt='%.0f,%.3f', header='ID,MEDV',delimiter=',',comments='') 


#read complete data
x=read_fulldata('data/train.csv')


#read the trg attributes
trg_attr=read_attr('data/train.csv')
#find max, min, mean and standard deviation value in data
max_val=np.max(trg_attr,axis=0)
min_val=np.min(trg_attr,axis=0)
calc_mean=np.mean(trg_attr,axis=0)
dev=np.std(trg_attr,axis=0)


#split the data into training and validation
#np.random.shuffle(x)
#training_idx = np.random.randint(x.shape[0], size=300)
#valid_idx = np.random.randint(x.shape[0], size=100)
#training, validation = x[training_idx,:], x[valid_idx,:]
#training, validation = x[:320,:], x[80:,:]

training=x
#training attributes
trg_attr=np.delete(training,[13],1)
#taking log of LSTAT feature/ attribute 
trg_attr=log_lstat(trg_attr)

#training y(MEDV values)
trg_y=get_column(training,13)
trg_y=np.array(trg_y)
trg_y=np.reshape(trg_y,(len(trg_y),1))


#trg_y=read_y('data/train.csv')

#normalise the data
norm_attr=normalize(trg_attr,calc_mean,dev)
#feautre engineering
phi=feature_man(norm_attr)
#remove outliners
outliners=remove_outliners(norm_attr)
phi=np.delete(phi,outliners,axis=0)
y=np.delete(trg_y,outliners,axis=0)

y=np.array(y)


#setting parameters
step_size = 0.0045 # step size
conv_condn=0.000001 #condition for convergence
max_iter=30000 #maximum number of iterations
lamda=0.0005 #value of lambda

#calculate w using different functions defined
#calculating closed form solution
w_linear=linear_reg(norm_attr,trg_y)
#calculating gradient descent for linear regression
print("calculating gradient descent for linear regression ")
w_grad=gradientdescent(phi,y,step_size,conv_condn,max_iter,lamda)
print("calculating gradient descent with lp norm =1.25 ")
#calculating gradient descent with lp norm =1.25
w_lpnorm1=gradientdescent_lp(phi,y,step_size,conv_condn,max_iter,lamda,1.25)
print("calculating gradient descent with lp norm =1.5 ")
#calculating gradient descent with lp norm =1.5
w_lpnorm2=gradientdescent_lp(phi,y,step_size,conv_condn,max_iter,lamda,1.5)
print("calculating gradient descent with lp norm =1.75 ")
#calculating gradient descent with lp norm =1.75
w_lpnorm3=gradientdescent_lp(phi,y,step_size,conv_condn,max_iter,lamda,1.75)

'''
--------------------------------------------------------------------------------------
			CROSS VALIDATION : not used for final submission
--------------------------------------------------------------------------------------
#printing RMS error
err2,LSS2=costfn(phi,w_grad,y)
print(LSS2)

err3,LSS3=costfn(phi,w_lpnorm1,y)
print(LSS3)

err4,LSS4=costfn(phi,w_lpnorm2,y)
print(LSS4)
err5,LSS5=costfn(phi,w_lpnorm3,y)
print(LSS5)

#checking with validation data Cross validation
valid_attr=np.delete(validation,[13],1)
#valid_attr=log_lstat(valid_attr)
valid_y=get_column(validation,13)
valid_y=np.array(valid_y)
valid_y=np.reshape(valid_y,(len(valid_y),1))

valid_attr=normalize(valid_attr,calc_mean,dev)
phi=feature_man(valid_attr)

#predicted_valid_y=np.matmul(phi,w_grad)

#for closed form solution
err1,LSS1=costfn(phi,w_linear,valid_y)
#for gradient descent algortihm
err2,LSS2=costfn(phi,w_grad,valid_y)
print(LSS1,LSS2)

#for lp_norm
err3,LSS3=costfn(phi,w_lpnorm1,valid_y)
print(LSS3)
err4,LSS4=costfn(phi,w_lpnorm2,valid_y)
print(LSS4)
err5,LSS5=costfn(phi,w_lpnorm3,valid_y)
print(LSS5)
-------------------------------------------------------------------------------------------------------
'''

#read attributes of test data
test_attr=read_attr("data/test.csv")
test_attr=log_lstat(test_attr)
#normalise the test attributes
test_attr=normalize(test_attr,calc_mean,dev)

#feature engineering
phi=feature_man(test_attr)

#writing output to csv files
writeoutput("data/test.csv",phi,w_grad,"output.csv")
print("Output for gradient descent written to file data/test.csv")
writeoutput("data/test.csv",phi,w_lpnorm1,"output_p1.csv")
print("Output written to file data/test.csv for gradient descent with lp norm =1.25 ")
writeoutput("data/test.csv",phi,w_lpnorm2,"output_p2.csv")
print("Output written to file data/test.csv for gradient descent with lp norm =1.5 ")
writeoutput("data/test.csv",phi,w_lpnorm3,"output_p3.csv")
print("Output written to file data/test.csv for gradient descent with lp norm =1.75 ")

