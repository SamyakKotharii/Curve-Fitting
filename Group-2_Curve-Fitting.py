'''
Group 2
Dhruvil Panchamia (AU1940285)
Jash Shah (AU1940286)
Samyak Kothari (AU1940211)
Hiral Shah (AU1940128)
'''

import matplotlib.pyplot as plt
import numpy as np
import os

def leastSquares():
    # Taking data points as input and inserting it into list
    lst1 = []
    n1 = int(input("Enter number of  Data Points: "))
    print("Enter X Co - ordinates: ")
    for i in range(0,n1):
        x = float(input())
        lst1.append(x)
    print("X Co - ordinates =",lst1)

    input("")

    lst2 = []
    n2 = n1
    print("Enter Y Co - ordinates: ")
    for j in range(0,n2):
        y = float(input())
        lst2.append(y)
    print("Y Co - ordinates =",lst2)

    input("")

    #plotting data points taken from user
    plt.plot(lst1,lst2,'o',color = 'black')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Data Points')
    plt.show()
    input("")

    # Generating Matrix X which has 2 columns and n rows (n = no. of data points) where column one has all 1's and column two has x co-ordinates of data points
    N1 = n1
    M1 = 2
    X=np.array([[1,lst1[k]] for k in range(N1)])
    print("X:")


    for i in X:

        for j in i:

            print(j, end=" ")

        print("")
    input("")

    # Generating Matrix Y which has 1 column and n rows containing y co-ordinates of data points
    N2 = n1
    M2 = 1
    Y = np.array([lst2[k] for k in range(N2)])
    print("Y:")
    for i in Y:   
        print(i)
    input("")

    # Doing transpose of X Matrix
    rez = np.array([[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]) 
    print("XT: ") 
    for row in rez: 
        print(row)
    input("")

    # Multilplying X Transpose with X
    res = np.array([[0 for x in range(2)] for y in range(2)])   
    for i in range(len(rez)): 
        for j in range(len(X[0])): 
            for k in range(len(X)):             
                res[i][j] += rez[i][k] * X[k][j]
    
    print("XT*X: ")
    for i in res:

        for j in i:

            print(j, end=" ")
            
        print("")
    input("")

    # Multiplying X Transpose with Y
    product = list()
    for i in range(2):
        row = rez[i]
        ele = 0
        for j in range(n1):
            ele += row[j]*Y[j]
        product.append(ele)
    print("XT*Y: ")
    for i in range(2):
        print(product[i])
    input("")

    
    # Doing Inverse of (XT*X)
    inv = [[0.0,0.0],[0.0,0.0]]
    inverse = res
    (inverse[0][0], inverse[1][1]) = (inverse[1][1], inverse[0][0])
    (inverse[0][1], inverse[1][0]) = (-inverse[1][0], -inverse[0][1])

    det = (res[0][0]*res[1][1])-(res[0][1]*res[1][0])
    inv[0][0] = float(inverse[0][0])/det
    inv[0][1] = float(inverse[0][1])/det
    inv[1][0] = float(inverse[1][0])/det
    inv[1][1] = float(inverse[1][1])/det
    print("(XT*X)^-1: ")
    print(inv)

    input("")

    # Multiplying (XT*X)^-1 with (XT*Y) and finding least square solution of data points
    ans = list()
    for i in range(2):
        row = inv[i]
        ele = 0
        for j in range(2):
            ele += row[j]*product[j]
        ans.append(ele)
    print("(XT*X)^-1*(XT*Y): ")
    for i in range(2):
        print(ans[i])
    input("")

    bo = ans[0]
    b1 = ans[1]
    bo = round(bo,2)
    b1 = round(b1,2)

    print("bo = ", bo)
    print("b1 = ", b1)
    input("")

    #Plotting best fit line to data points
    x = np.linspace(-5,10,100)
    y = bo*x + b1
    plt.plot(lst1,lst2,'o',color = 'black')
    plt.plot(x, y,'-r')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Curve Fitting Using Least Square')
    plt.show()
    input("")

    #for finding Goodness of Fit using correlation coefficient
    print("**Goodness of fit**")
    totalX=0 
    sumY=0
    for ele in range(0, len(lst1)): 
        totalX = totalX + lst1[ele] 
    # printing sum of all X co-ordinates
    print("\nSum of all elements in X co-ordinates: ", totalX)

    for ele in range(0, len(lst2)): 
        sumY = sumY + lst2[ele] 
    # printing sum of all Y co-ordinates 
    print("\nSum of all elements in Y co-ordinates: ", sumY)
    products= []
    sumofproducts=0
    for num1, num2 in zip(lst1, lst2):
        products.append(num1 * num2)
        
    for ele in range(0, len(products)): 
        sumofproducts = sumofproducts + products[ele] 
    # printing sum of product of corresponding x and y co-efficient  
    print("\nSum of all corresponding x*y : ", sumofproducts)

    
    sumofsqx=0
    sumofsqy=0
    
    for i in lst1:
        sumofsqx+=i*i;
    #printing sum of square of all X co-ordinates
    print("\nSum of squares of elements in X : ",sumofsqx)
    for i in lst2:
        sumofsqy+=i*i;
    #printing sum of square of all Y co-ordinates
    print("\nSum of squares of elements in Y : ",sumofsqy)

    sqrX= (totalX)*(totalX)
    sqrY= (sumY)*(sumY)
    #printing Square of Sum of all Y co-ordinates:
    print("\nSquare of Sum of all Y co-ordinates:",sqrY)
    #printing Square of Sum of all X co-ordinates:
    print("\nSquare of Sum of all X co-ordinates:",sqrX)
    m = (n1*sumofsqx) - sqrX
    n = (n1*sumofsqy) - sqrY
    o=m*n
    sqrt = np.sqrt(o)
    
    a = (n1*sumofproducts) - ((sumY)*(totalX))
    #Correlation Coefficient(r)
    r = ( a / sqrt)
    #printing correlation co-efficient
    print("\nCorrelation Coefficient r = ",r)

    #printing the Goodness of fit of least square
    if r==1 or r==-1 :
        print("\n***Perfect Fit***")
    elif 0.5 <= r < 1 or -0.5 >= r >-1 :
        print("\n***Good Fit***")
    elif 0 < r < 0.5 or 0 > r > -0.5:
        print("\n***Not a good fit***")
    else:
        print("\n***No relationship between the two variables***")

    lsty=[]
    for j in range(0,len(lst1)):
        x=lst1[j]
        y=0
        y=bo+b1*x
        lsty.append(y)
  
    print("The values of y from the equation corresponding to the input x is : ",*lsty, sep = ", ")

    #for root mean square error
    sum=float(0)
    for i in range(0,len(lst2)):
        sum = sum + np.square(lst2[i]- lsty[i])
    e=float(0)
    e=(1/n1)*sum
    RootMeanSqr=float(0)
    RootMeanSqr=np.sqrt(e)
    #printing root mean square error
    print("\nRoot Mean Square Error =",RootMeanSqr)
    input("")
# plotting using QR Decomposition and least square
def qr():
    # Taking data points as input and inserting it into list
    lst1 = []
    n1 = int(input("Enter number of  Data Points: "))
    print("Enter X Co - ordinates: ")
    for i in range(0,n1):
        x = float(input())
        lst1.append(x)
    print("X Co - ordinates =",lst1)

    input("")

    lst2 = []
    n2 = n1
    print("Enter Y Co - ordinates: ")
    for j in range(0,n2):
        y = float(input())
        lst2.append(y)
    print("Y Co - ordinates =",lst2)

    input("")

    #plotting data points taken from user
    plt.plot(lst1,lst2,'o',color = 'black')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Data Points')
    plt.show()
    input("")

    # Generating Matrix X which has 2 columns and n rows (n = no. of data points) where column one has all 1's and column two has x co-ordinates of data points
    N1 = n1
    M1 = 2
    X=np.array([[1,lst1[k]] for k in range(N1)])
    print("X:")


    for i in X:

        for j in i:

            print(j, end=" ")

        print("")
    input("")

    # Generating Matrix Y which has 1 column and n rows containing y co-ordinates of data points
    N2 = n1
    M2 = 1
    Y = np.array([lst2[k] for k in range(N2)])
    print("Y:")
    for i in Y:   
        print(i)
    input("")

    # Taking X1 as first column of a matrix X
    print("X1: ")
    x1 = np.array([[1] for k in range(N1)])
    print(x1)
    input("")

    # Taking X2 as second column of a matrix X
    print("X2: ")
    x2 = np.array([lst1[k] for k in range(N1)])
    print(x2)
    input("")

    # Taking V1=X1
    print("V1: ")
    v1 = x1
    print(v1)
    input("")

    # Multiplying X2 vector with V1 vector
    aaa = 0
    for i in range(N1):
                  aaa += x2[i]*v1[i]

    # Multiplying V1 vector with V1 vector
    bbb = 0
    for i in range(N1):
        bbb += v1[i]*v1[i]

    # Doing (X2*V1)/V1*V1
    ccc = aaa/bbb

    # Doing ((X2*V1)/(V1*V1))*X1
    z2 = ccc*x1

    # Finding V2 = X2 - ((X2*V1)/(V1*V1))*X1
    print("V2: ")
    lst3 = []
    for i in range(n1):
        v2 = x2[i] - z2[i]
        lst3.append(v2)
    print(lst3)
    input("")

    # Finding unit vector V1 
    print("V1': ")
    lst4 = []
    abc = bbb**0.5
    for i in range(N1):
        unit1 = v1[i]/abc
        lst4.append(unit1)
    print(lst4)
    input("")

    # Findind V2*V2
    pqr=0
    for i in range(N1):
        pqr += lst3[i]*lst3[i]

    # Findinfg unit vector of V2
    print("V2': ")
    lst5 = []
    xyz = pqr**0.5
    for i in range(N1):
        unit2 = lst3[i]/xyz
        lst5.append(unit2)
    print(lst5)
    input("")

    # Taking Q as a n*2 matrix where n is number of data points where first column is unit vector of V1 and second column is unit vector of V2
    Q = np.array([[lst4[k],lst5[k]] for k in range(N1)])
    print("Q: ")
    for i in Q:

        for j in i:

            print(j, end=" ")

        print("")
    input("")    

    # Finding transpose of matrix Q
    rez = np.array([[Q[j][i] for j in range(len(Q))] for i in range(len(Q[0]))]) 
    print("QT: ") 
    for row in rez: 
        for i in range(row.shape[0]):
            print(row[i],end = " ")
        print(" ")
    input("")

    # Finding R = QT*X
    R = [[0.0,0.0],[0.0,0.0]]
    R = np.array([[0.0 for x in range(2)] for y in range(2)])   
    for i in range(len(rez)): 
        for j in range(len(X[0])): 
            for k in range(len(X)):             
                R[i][j] += rez[i][k] * X[k][j]

    print("QT*X: ")
    for i in R:

        for j in i:

            print(j, end=" ")
            
        print("")
    input("")

    # Finding inverse of R
    inv = [[0.0,0.0],[0.0,0.0]]
    inverse = R
    (inverse[0][0], inverse[1][1]) = (inverse[1][1], inverse[0][0])
    (inverse[0][1], inverse[1][0]) = (-inverse[1][0], -inverse[0][1])

    det = (R[0][0]*R[1][1])-(R[0][1]*R[1][0])
    inv[0][0] = float(inverse[0][0])/det
    inv[0][1] = float(inverse[0][1])/det
    inv[1][0] = float(inverse[1][0])/det
    inv[1][1] = float(inverse[1][1])/det
    print("(R)^-1: ")
    print(inv)
    input("")

    # Finding XT*Y
    product = list()
    for i in range(2):
        row = rez[i]
        ele = 0
        for j in range(n1):
            ele += row[j]*Y[j]
        product.append(ele)
    print("QT*Y: ")
    for i in range(2):
        print(product[i])
    input("")

    # Finding least square solution b = R^-1*QT*Y
    ans = list()
    for i in range(2):
        row = inv[i]
        ele = 0
        for j in range(2):
            ele += row[j]*product[j]
        ans.append(ele)
    print("R^-1*QT*Y: ")
    for i in range(2):
        print(ans[i])
    input("")

    bo = ans[0]
    b1 = ans[1]
    #bo = round(bo,2)
    #b1 = round(b1,2)

    print("bo = ", bo)
    print("b1 = ", b1)
    input("")

    #Plotting best fit line to data points
    x = np.linspace(-5,10,100)
    y = bo*x + b1
    plt.plot(lst1,lst2,'o',color = 'Blue')
    plt.plot(x, y,'-r')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Curve Fitting QR Decomposition')
    plt.show()
    input("")
    

    #for finding Goodness of Fit using correlation coefficient
    print("**Goodness of fit**")
    totalX=0 
    sumY=0
    for ele in range(0, len(lst1)): 
        totalX = totalX + lst1[ele] 
    # printing sum of all X co-ordinates
    print("\nSum of all elements in X co-ordinates: ", totalX)

    for ele in range(0, len(lst2)): 
        sumY = sumY + lst2[ele] 
    # printing sum of all Y co-ordinates 
    print("\nSum of all elements in Y co-ordinates: ", sumY)
    products= []
    sumofproducts=0
    for num1, num2 in zip(lst1, lst2):
        products.append(num1 * num2)
        
    for ele in range(0, len(products)): 
        sumofproducts = sumofproducts + products[ele] 
    # printing sum of product of corresponding x and y co-efficient  
    print("\nSum of all corresponding x*y : ", sumofproducts)

    
    sumofsqx=0
    sumofsqy=0
    
    for i in lst1:
        sumofsqx+=i*i;
    #printing sum of square of all X co-ordinates
    print("\nSum of squares of elements in X : ",sumofsqx)
    for i in lst2:
        sumofsqy+=i*i;
    #printing sum of square of all Y co-ordinates
    print("\nSum of squares of elements in Y : ",sumofsqy)

    sqrX= (totalX)*(totalX)
    sqrY= (sumY)*(sumY)
    #printing Square of Sum of all Y co-ordinates:
    print("\nSquare of Sum of all Y co-ordinates:",sqrY)
    #printing Square of Sum of all X co-ordinates:
    print("\nSquare of Sum of all X co-ordinates:",sqrX)
    m = (n1*sumofsqx) - sqrX
    n = (n1*sumofsqy) - sqrY
    o=m*n
    sqrt = np.sqrt(o)
    
    a = (n1*sumofproducts) - ((sumY)*(totalX))
    #Correlation Coefficient(r)
    r = ( a / sqrt)
    #printing correlation co-efficient
    print("\nCorrelation Coefficient r = ",r)
    #printing the Goodness of fit of least square
    if r==1 or r==-1 :
        print("\n***Perfect Fit***")
    elif 0.5 <= r < 1 or -0.5 >= r >-1 :
        print("\n***Good Fit***")
    elif 0 < r < 0.5 or 0 > r > -0.5:
        print("\n***Not a good fit***")
    else:
        print("\n***No relationship between the two variables***")
    lsty=[]
    for j in range(0,len(lst1)):
        x=lst1[j]
        y=0
        y=bo+b1*x
        lsty.append(y)

    #for root mean square error
    sum=float(0)
    for i in range(0,len(lst2)):
        sum = sum + np.square(lst2[i]- lsty[i])
    e=float(0)
    e=(1/n1)*sum
    RootMeanSqr=float(0)
    RootMeanSqr=np.sqrt(e)
    #printing root mean square error
    print("\nRoot Mean Square Error =",RootMeanSqr)
    input("")

    
def polynomial():
    lst1 = []
    lst2 = []

    #function to define cofactor
    def cofactor(m, i, j):
        return [row[: j] + row[j+1:] for row in (m[: i] + m[i+1:])]
        
    #function to find the determinant of a matrix
    def matrixDeterminant(matrix):
     
       #for 2*2 matrix
        if(len(matrix) == 2):
            value = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
            return value
     
        Sum = 0
     
       #for n*n matrix
        for currentcolumn in range(len(matrix)):
     
            sign = pow(-1,currentcolumn)
     
          
            subdeterminant = matrixDeterminant(cofactor(matrix, 0, currentcolumn))
     
           
            Sum += (sign * matrix[0][currentcolumn] * subdeterminant)
     
        # returning the determinant value
        return Sum
     
        print("The determinant of the matrix is:", matrixDeterminant(matrix))
        

    # Taking data points as input and inserting it into list
    print("For polynomial fit atleast 3 data points are required.")
    n1 = int(input("Enter number of  Data Points: "))
    print("Enter X Co - ordinates: ")
    for i in range(0,n1):
        x = float(input())
        lst1.append(x)
    print("X Co - ordinates =",lst1)

    input("")

        
    n2 = n1
    print("Enter Y Co - ordinates: ")
    for j in range(0,n2):
        y = float(input())
        lst2.append(y)
    print("Y Co - ordinates =",lst2)

    input("")

    #plotting data points taken from user
    plt.plot(lst1,lst2,'o',color = 'black')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Data Points')
    plt.show()
    input("")
        
    #solving for polynomial coefficients using cramer's rule
    print("Enter the degree of the polynomial in the range 2,",n1)
    n=int(input("Enter the degree for the polynomial: "))
        
    #Matrix M
       
    M=[]
    for i in range(0,n+1):
        a=[]
        for j in range(0,n+1):
            sum=0
            for k in range(0,len(lst1)):
                    
                ele=lst1[k]
                power=i+j
                sum=sum+pow(ele,power)
            a.append(sum)
        M.append(a)
    print("Matrix M:")
    

    print(np.matrix(M))
    detM = matrixDeterminant(M)
    input("")
    print("The determinant of matrix M is: ",detM)
    input("")
    b=[]
    for i in range(0,n+1):
        a=[]
        for j in range(0,1):
            sum=0
            for k in range(0,len(lst1)):
                y=lst2[k]
                x=lst1[k]
                power=i
                sum=sum+ pow(x,power)*y
            a.append(sum)
        b.append(a)
    print("Matrix b:")
    print(np.matrix(b))
    input("")


      
    determinant=[]
    X=M
    for j in range(0,n+1):
        
        
        for i in range(0,n+1):
            X[i][j]=b[i][0]
            
            
                  
        print("Matrix M",j)
        print(np.matrix(X))
        input("")
        a=matrixDeterminant(X)
        print("The determinant of Matrix M",j,"is: ",a)
        input("")
       
        determinant.append(a)
        X=[]
        for i in range(0,n+1):
            
            a=[]
            for j in range(0,n+1):
                sum=0
                for k in range(0,len(lst1)):
                    
                    ele=lst1[k]
                    power=i+j
                    sum=sum+pow(ele,power)
                a.append(sum)
            X.append(a)
    print("The list of determinants of matrices are: ",determinant)
    input("")   
        
    coefficients=[]
    for i in range(0,n+1):
        det=determinant[i]
        c=det/detM
        coefficients.append(c)
    print("The list of coefficients is: ",coefficients)
    input("")   
    #polynomial will be in the form "y=coefficients[n]*x^n + ....... + coefficients[0]*x^0"

    yx=[]
    for j in range(0,len(lst1)):
        x=lst1[j]
        y=0
        for i in range (0,n+1):
            
        
            y=y+coefficients[i]*pow(x,i)
        yx.append(y)
    print("The values of y from the equation corresponding to the input x is: ",yx)
    input("")
    y=0
    x=np.linspace(-5,10,100)
    for i in range (0,n+1):
        
        
        y=y+coefficients[i]*pow(x,i)
            
    #ploting the data points and the polynomial
    plt.plot(lst1,lst2,'o',color = 'black')
    plt.plot(x,y,'-r')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Data Points and Polynomial Fit')
    plt.show()
    input("")
    
    #for finding Goodness of Fit using correlation coefficient
    print("**Goodness of fit**")
    totalX=0 
    sumY=0
    for ele in range(0, len(lst1)): 
        totalX = totalX + lst1[ele] 
    # printing sum of all X co-ordinates
    print("\nSum of all elements in X co-ordinates: ", totalX)

    for ele in range(0, len(lst2)): 
        sumY = sumY + lst2[ele] 
    # printing sum of all Y co-ordinates 
    print("\nSum of all elements in Y co-ordinates: ", sumY)
    products= []
    sumofproducts=0
    for num1, num2 in zip(lst1, lst2):
        products.append(num1 * num2)
        
    for ele in range(0, len(products)): 
        sumofproducts = sumofproducts + products[ele] 
    # printing sum of product of corresponding x and y co-efficient  
    print("\nSum of all corresponding x*y : ", sumofproducts)

    
    sumofsqx=0
    sumofsqy=0
    
    for i in lst1:
        sumofsqx+=i*i;
    #printing sum of square of all X co-ordinates
    print("\nSum of squares of elements in X : ",sumofsqx)
    for i in lst2:
        sumofsqy+=i*i;
    #printing sum of square of all Y co-ordinates
    print("\nSum of squares of elements in Y : ",sumofsqy)

    sqrX= (totalX)*(totalX)
    sqrY= (sumY)*(sumY)
    #printing Square of Sum of all Y co-ordinates:
    print("\nSquare of Sum of all Y co-ordinates:",sqrY)
    #printing Square of Sum of all X co-ordinates:
    print("\nSquare of Sum of all X co-ordinates:",sqrX)
    m = (n1*sumofsqx) - sqrX
    n = (n1*sumofsqy) - sqrY
    o=m*n
    sqrt = np.sqrt(o)
    
    a = (n1*sumofproducts) - ((sumY)*(totalX))
    #Correlation Coefficient(r)
    r = ( a / sqrt)
    #printing correlation co-efficient
    print("\nCorrelation Coefficient r = ",r)
    #printing the Goodness of fit of least square
    if r==1 or r==-1 :
        print("\n***Perfect Fit***")
    elif 0.5 <= r < 1 or -0.5 >= r >-1 :
        print("\n***Good Fit***")
    elif 0 < r < 0.5 or 0 > r > -0.5:
        print("\n***Not a good fit***")
    else:
        print("\n***No relationship between the two variables***")
    

    
   
    sum=float(0)
    #for value of Root Mean Square
    for i in range(0,len(lst2)):
        sum = sum + np.square(lst2[i]- yx[i])
    e=float(0)
    e=(1/n1)*sum
    RootMeanSqr=float(0)
    RootMeanSqr=np.sqrt(e)
    #printing root mean square error
    print("\nRoot Mean Square Error = ",RootMeanSqr)
    input("")

while(True):
    os.system('CLS')
    
    print ("---------------------------------Curve Fitting---------------------------------")
    print("")

    print("1. Linear Curve Fitting")
    print("2. Polynomial Curve Fitting")
    print("3. Exit")
    choice=int(input("Enter a choice: "))
    os.system('CLS')

    if (choice == 1):
        print("------------------------------Linear Curve Fitting------------------------------")
        print("")
        print("1. Using Least Square Method")
        print("2. Using QR Decomposition method")
        print("")
        option=int(input("Choose any Method: "))
        os.system('CLS')
        if(option==1):
            print("\t\t\t\tLeast Square Method")
            print("")
            leastSquares()
        elif(option == 2):
            print("\t\t\t\tQR Decomposition Method")
            print("")
            qr()
        else:
            print("Choose a correct option!")
    elif(choice == 2):
        print("----------------------------Polynomial Curve Fitting----------------------------")
        print("")
        polynomial()
    elif(choice == 3):
        break
    else:
        print("Choose a correct option!")




