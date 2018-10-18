class Employee(object):
    """docstring for Employee."""
    def __init__(self,First_Name,Second_Name,Salary):
            self.First_Name=First_Name
            self.Second_Name=Second_Name
            self.Salary=Salary
    def method(self):
        return '{},{}'.format(self.First_Name, self.First_Name)

emp_1=Employee('Alejandro','Soumah','1.50')
