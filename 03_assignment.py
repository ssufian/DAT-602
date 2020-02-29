

'''
Assignment #3
1. Add / modify code ONLY between the marked areas (i.e. "Place code below"). Do not modify or add code elsewhere.
2. Run the associated test harness for a basic check on completeness. A successful run of the test cases does not guarantee accuracy or fulfillment of the requirements. Please do not submit your work if test cases fail.
3. To run unit tests simply use the below command after filling in all of the code:
    python 03_assignment.py
4. Unless explicitly stated, please do not import any additional libraries but feel free to use built-in Python packages
5. Submissions must be a Python file and not a notebook file (i.e *.ipynb)
6. Do not use global variables
7. Use the test cases to infer requirements wherever feasible - not all exercises will have test cases
'''
import csv, json, math, requests, unittest, uuid, pandas as pd

# ------ Create your classes here \/ \/ \/ ------

#BankAccount class declaration below here

class BankAccount():   
        
    """A simple attempt to represent a bank account."""

    def __init__(self, bankname, firstname, lastname):        
        """Initialize attributes to describe a bank account."""        
        self.bankname = bankname        
        self.firstname = firstname        
        self.lastname = lastname
        self.balance = 0
           
    def read_BankAccount(self):        
        """Print a statement showing the bank account current balance."""        
        print("This bank account has " + str(self.balance) + " dollars in it.")

    def deposit(self, amount):        
        """To deposit money."""        
        self.balance = self.balance + amount
        print("This bank account has " + str(self.balance) + " dollars in it.")
        
    def withdrawal(self, amount):        
        """To withdraw money."""        
        if amount <= self.balance:
            self.balance = self.balance - amount
            print("This bank account has " + str(self.balance) + " dollars in it.")
        else:
            print('You are overdrawn')
            
    def __str__(self):
        return 'Bank name = {self.bankname}, Firstname = {self.firstname}, Lastname = {self.lastname}, Balance = {self.balance}'.format(self=self)
        
   
# Box class declaration below here

class Box():

    """A simple attempt to represent a Box."""

    def __init__(self, length, width):        
        """Initialize attributes to describe length & width of box."""        
        self.length = length        
        self.width = width        
        print ('This box has length = {self.length} & width = {self.width}'.format(self=self))
    
    def __str__(self):
          return 'Area of Box = {self.area}, Perimeter of Box = {self.perimeter}'.format(self=self)
        
    def render(self):
        """Prints out to the screen a box made with asterisks of length and width dimensions.""" 
        for i in range(1, self.width+1) : 
            for j in range(1, self.length+1) : 
                if (i == 1 or i == self.width or
                    j == 1 or j == self.length) : 
                    print("*",end='')             
                else : 
                    print(" ",end='')               
            print() 
  
    def invert(self):
        """switches length and width with each other"""
        for i in range(1, self.length+1) : 
            for j in range(1, self.width+1) : 
                if (i == 1 or i == self.length or
                    j == 1 or j == self.width) : 
                    print("*",end='')             
                else : 
                    print(" ",end='')               
            print()        
        
    def get_area(self):
        """Compute area of box"""
        self.area = self.length*self.width
        return self.area
        
    def get_perimeter(self):
        """Compute perimeter of box"""
        self.perimeter = 2*(self.length  +  self.width)
        return self.perimeter       
        
    def double(self):
        """Double the box size"""
        self.length = self.length*2
        self.width = self.width*2
        return(self)
        
    
    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.width == other.width and self.length == other.length
        return False
    
    def print_dim(self):
        """print specific length and width of box"""
        print("The length of box is {self.length}".format(self=self))
        print("The width of box is {self.width}".format(self=self))
        
    def get_dim(self):
        """print specific length and width of box in a tuple""" 
        return (self.length, self.width)
    
    def combine(self,other):
        """takes another box as an argument and increases the length and width""" 
        """by the dimensions of the box passed in"""
        return (Box(self.length + other.length, self.width + other.width))
        
    def get_hypot(self):
        """finds the length of the diagonal that cuts throught the middle"""
        self.hypo = int(round((self.length**2 +self.width**2)**0.5,0))
        return self.hypo

    def get_length(self):
        """gets length of box"""
        return int(self.length)

    def get_width(self):
        """gets width of box"""
        return int(self.width)



    


    
# ------ Create your classes here /\ /\ /\ ------




def exercise01():
    '''
        Create a class called BankAccount that has four attributes: bankname, firstname, lastname, and balance. 
        The default balance should be set to 0.  (Create your class above.)

        n addition, create ...
        - A method called depost() that allows the user to make deposts into their balance. 
        - A method called withdrawal() that allows the user to withdraw from their balance. 
        - Withdrawls may not exceed the available balance.  Hint: consider a conditional argument in your withdrawl() method.
        - Use the __str__() method in order to display the bank name, owner name, and current balance.

        In the function of exercise01():
        - Make a series of deposts and withdraws to test your class (below).

'''

    # ------ Place code below here \/ \/ \/ ------

    my_bankacct =  BankAccount('Bank of England','Johnny my man', 'Rocket')

    my_bankacct.deposit(120)
    my_bankacct.deposit(120)
    my_bankacct.deposit(120)

    my_bankacct.withdrawal(60)
    my_bankacct.withdrawal(25)

    my_bankacct.read_BankAccount()

    print(my_bankacct)



    # ------ Place code above here /\ /\ /\ ------





def exercise02():

    '''
        Create a class Box that has attributes length and width that takes values for length and width upon construction (instantiation via the constructor). 
        Make sure to use Python 3 semantics. 
       
        In addition, create...
        - A method called render() that prints out to the screen a box made with asterisks of length and width dimensions
        - A method called invert() that switches length and width with each other
        - Methods get_area() and get_perimeter() that return appropriate geometric calculations
        - A method called double() that doubles the size of the box. Hint: Pay attention to return value here
        - Implement __eq__ so that two boxes can be compared using ==. Two boxes are equal if their respective lengths and widths are identical.
        - A method print_dim that prints to screen the length and width details of the box
        - A method get_dim that returns a tuple containing the length and width of the box
        - A method combine() that takes another box as an argument and increases the length and width by the dimensions of the box passed in
        - A method get_hypot() that finds the length of the diagonal that cuts throught the middle
        
        In the function exercise02():
        - Instantiate 3 boxes of dimensions 5,10 , 3,4 and 5,10 and assign to variables box1, box2 and box3 respectively 
        - Print dimension info for each using print_dim()
        - Evaluate if box1 == box2, and also evaluate if box1 == box3, print True or False to the screen accordingly
        - Combine box3 into box1 (i.e. box1.combine())
        - Double the size of box2
        - Combine box2 into box1
        - Using a for loop, iterate through and print the tuple received from calling box2's get_dim()
        - Find the size of the diagonal of box2
'''

    # ------ Place code below here \/ \/ \/ ------

    b1=Box(16,28)
    b2=Box(6,8)
    b3=Box(5,10)
    return b1, b2,b3

    b1.print_dim()
    b2.print_dim()
    b3.print_dim()
    return b1, b2,b3

    print(b1 == b2)
    print(b1 == b3)  

    b1.combine(b3)
    b1.combine(b2)

    b2.double()
    return b2

    for i in b2.get_dim():
        print (i)

    b2.get_hypot()
    return b2

    

  
    # ------ Place code above here /\ /\ /\ ------





def exercise03():
    '''
    1. Read about avocado prices on Kaggle (https://www.kaggle.com/neuromusic/avocado-prices/home)
    2. Load the included avocado.csv file and display every line to the screen
    3. Use the imported csv library
    '''

    # ------ Place code below here \/ \/ \/ ------
  
    pd.set_option('display.max_rows', None)

    df = pd.read_csv("avocado.csv",index_col=0) 

    df.head(10) #see whats the original file looks like without index

    df.columns = ['Date', 'Average_Price' ,'Total_volume','PLU_4046_Sold','PLU_4225_Sold','PLU_4770_Sold','Total_Bags', 'Small_Bags', 'Large_Bags','XLarge_Bags','type','year','region']
    df.head(10) #see whats like after column renaming


    print(df)  

    
    # ------ Place code above here /\ /\ /\ ------


class TestAssignment3(unittest.TestCase):

    def test_exercise02(self):
        print('Testing exercise 2')
        b1, b2, b3 = exercise02()
        self.assertEqual(b1.get_length(),16)
        self.assertEqual(b1.get_width(),28)
        self.assertTrue(b1==Box(16,28))
        self.assertEqual(b2.get_length(),6)
        self.assertEqual(b2.get_width(),8)
        self.assertEqual(b3.get_length(),5)
        self.assertEqual(b2.get_hypot(),10)
        self.assertEqual(b1.double().get_length(),32)
        self.assertEqual(b1.double().get_width(),112)
        self.assertTrue(6 in b2.get_dim())
        self.assertTrue(8 in b2.get_dim())
        self.assertTrue(b2.combine(Box(1,1))==Box(7,9))

        
    def test_exercise03(self):
        print('Exercise 3 not tested')
        exercise03()
     

if __name__ == '__main__':
    unittest.main()
