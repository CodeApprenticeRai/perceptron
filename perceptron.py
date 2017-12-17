
# coding: utf-8



import numpy as np


class perceptron():
    
	#Initialize with inputs already already set. 
	#This will be changed in order to accept inputs 
    def __init__(self):
		
        self.x = np.array([[0,0,0],
						   [1,0,0],
						   [1,1,0],
						   [1,1,1],
						   [0,1,1],
						   [1,0,1],
						   [0,1,0]])
			
		#If the third input is a 0, the output is a zero
		self.y = np.array([[0],
						   [0],
						   [0],
						   [1],
						   [1],
						   [1],
						   [0]])
        
        np.random.seed(0)
        self.w1 = np.random.random((3,1))
    
	#turns transformed variable into a probability, i.e. 0 > sigmoid(tv) < 1
    def sigmoid(self, tv, deriv=False):
        if deriv:
            return tv * (1-tv)
        
        return 1/(1+np.exp(-tv))
    
	#Tries to minimize error between our guess (y_hat), and the true values (y) .
	
	#Contains bugs
    def train(self):
        
        for i in range(60000):
            layer0 = self.x
            layer1 = self.sigmoid(np.dot(self.x,self.w1))
            self.y_hat = layer1

            self.error = self.y - self.y_hat

            if (i % 1000 == 0):
                print("The error at {} iterations is:".format(i), self.error,sep="\n")
            
            #play around with this 
            self.d_w1 = np.dot( self.y.T, ( self.error * self.sigmoid(self.y_hat, deriv=True) ) )
          
            self.w1 += self.d_w1 
        
        print("")
        print("The Post-Trained Error is: {}",self.error,sep="\n")
        print("The Post-Trained Weights are:", self.w1, sep="\n")
        return
        
        
       
    def predict(self, x):
        return self.sigmoid(np.dot(x,self.w1))
    
