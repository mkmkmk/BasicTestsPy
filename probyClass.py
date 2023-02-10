"""
 https://docs.python.org/3/tutorial/classes.html
"""

class Mutuj1:
	""" tu leci dokumentacja klasy """

	def __init__(self):
		self.pole = 3

		
k1 = Mutuj1()
k2 = k1
k2.pole = 30


def mutuj(mt: Mutuj1):
	mt.pole = 300


print(k1.pole)

mutuj(k2)

print(k1.pole)

print(Mutuj1.__doc__)

print("done")

#----------------------------------------------


#----------------------------------------------
class MyClass2:

	def __init__(self):
		self.atr1 = 1

	def nie_metoda():
		print("ok")

	def metoda(self):

		print("ok")
	
	def tez_metoda(this):

		this.atr1 = 2 
	
	
x = MyClass2()
MyClass2.nie_metoda() 
x.metoda()
MyClass2.metoda(x)

x.tez_metoda()
assert(x.atr1 == 2)

x.__dict__

x.metoda.__self__
x.metoda.__func__

#----------------------------------------------
def reverse(data):
	for index in range(len(data) - 1, -1, -1):
		yield data[index]
print("".join(reverse("nohtyp")))        


#----------------------------------------------






