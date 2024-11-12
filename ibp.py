import inspect

from collections import defaultdict
from functools import wraps
import sympy as sp
from sympy import latex

sp.init_printing(use_unicode=True)

def flatten(lst):
    return [item for sublist in lst for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]

def symmetry_policy(tuple):
    if tuple[0] == 0:
      return (tuple[1], tuple[0])
    else:
      return tuple

def check_input(func):
  '''
  This decorator checks that the type of the parameters passed
  to a function match the types declared in the definition
  of the function through python typing.

  The decorator relies on the `inspect` module, which exploits
  python reflection properties.
  '''
  @wraps(func)
  def wrapper(*args, **kwargs):
    # Retrieve function's signature
    signature = inspect.signature(func)
    # Create mapping from positional (args) and
    # keyword arguments (kwargs) to parameters
    bound_args = signature.bind(*args, **kwargs)

    # Set default values in case of missing arguments 
    bound_args.apply_defaults()

    # Now check consistency for types
    for arg in bound_args.arguments.items():
      declared_type = signature.parameters[arg[0]].annotation
      if declared_type is not inspect._empty and not isinstance(arg[1], declared_type):
        raise TypeError(f"Argument {arg[0]} is expected to be of type {declared_type.__name__}, "
                        f"but got {type(arg[1]).__name__}")
      
    
    return func(*args, **kwargs)

  return wrapper


m = sp.symbols('m')
D = sp.symbols('D')
p = sp.symbols('p')
p2 = p*p
m2 = m**2

# Master integrals
I11 = sp.symbols('I_{11}(D)')
I20 = sp.symbols('I_{20}(D)')
I02 = I20

master_integrals_map = {
  (1,1) : I11,
  (2,0) : I20,
  (0,2) : I02,
}

class BubbleIntegral:

  @check_input
  def __init__(self, nu1 : int,  nu2 : int):
    self.nu1 = nu1
    self.nu2 = nu2
    self.complexity = self.nu1 + self.nu2
    if self.complexity == 1:
      raise ValueError("Complexity must be greater than 1.")
    self.master_integral = True if self.complexity == 2 else False
    self.reduced = False
    self.__build_coefficients()


  def reduce(self):
    reduced_list = self.__reduce()

    tmp_dict = defaultdict(list)
    for e in reduced_list:
      tmp_dict[symmetry_policy(e[0])].append(e[1].simplify(rational=True))
    
    tmp_dict = [(key, sum(val).simplify()) for key, val in tmp_dict.items()]
    self.coefficients = tmp_dict
    self.reduced = True


  def __reduce(self):
    if not self.master_integral:
      empty_dict = {}
      for idx, coeff in enumerate(self.coefficients):
        nu1 = coeff[0][0]
        nu2 = coeff[0][1]
        if not nu1 + nu2 == 2:
          bubble_red = BubbleIntegral(nu1, nu2)
          # TODO
          tmp_list = bubble_red.__reduce()
          tmp_list = [(e[0],coeff[1] * e[1]) for e in tmp_list]
          empty_dict[idx] = tmp_list
      
          for idx, coeff_list in empty_dict.items():
            self.coefficients[idx] = coeff_list

      # Flatten list
      self.coefficients = flatten(self.coefficients)
      return self.coefficients
    else:
      return self.coefficients
  
  
  @classmethod
  def __construct_c1(cls):
    return - ( 2.0 * m2 - p2 ) / (p2 * (4*m2 - p2) )
  
  @classmethod
  def __construct_c2(cls, nu1, nu2):
    return 2.0 * nu2 / (nu1 - 1) * m2  / (p2 * (4*m2 - p2) )
  
  @classmethod
  def __construct_c3(cls, nu1, nu2):
    return - 1.0 / (nu1 - 1) * (D * p2 - (nu1 - 1) * (2.0 * m2 + p2) + 2 * nu2 * (m2 - p2))  / (p2 * (4*m2 - p2) )
  
  @classmethod
  def __tadpole_obp(cls, nu):
    '''
    Implementation of the IBP identity for :math:`\nu_1 \neq 0`, 
    :math:`\nu_2 = 0` according to the following relation

    .. math::

      I_{\nu_1, 0} = - \frac{D - 2 (\nu_1 - 1)}{2 \nu_1 m^2} I_{\nu_1-1,0}

    '''
    if not nu > 1:
      raise ValueError(f'The power of the propagator must be greater '
                       f'than 1 for the tadpole reduction, but got {nu}.')
    return - (D - 2 * (nu - 1)) / (2 * m2 * (nu - 1))
  
  def __build_coefficients(self):
    if self.master_integral:
      self.coefficients = [
        ((self.nu1, self.nu2), 1)
      ]
      return 
    
    if (self.nu1 == 0) ^ (self.nu2 == 0):
      nu = self.nu1 + self.nu2
      self.coefficients = [
        ((nu-1, 0), self.__tadpole_obp(nu))
      ]
      return
    else:
      if self.nu1 >= self.nu2:
        nu1 = self.nu1
        nu2 = self.nu2
      else:
        nu2 = self.nu1
        nu1 = self.nu2

      self.coefficients = [
        ((nu1, nu2 - 1), self.__construct_c1()),
        ((nu1 - 2, nu2 + 1), self.__construct_c2(nu1, nu2)),
        ((nu1 - 1, nu2), self.__construct_c3(nu1, nu2))
      ]
      return

  def get_coefficients(self):
    return self.coefficients
  
  def is_master_integral(self):
    return self.master_integral
  
  def is_reduced(self):
    return self.reduced
  

  def __rmul__(self, scalar):
    '''Override scalar * self'''
    self.__set_coefficient(scalar)
  
  def __mul__(self, scalar):
    '''Override self * scalar'''
    self.__rmul__(scalar)

  def show(self):
    if self.reduced:
      I_bub = sp.symbols(f'I_{self.nu1,self.nu2}(D)')
      aux = 0
      for coeff in self.coefficients:
        I_mi = sp.symbols(f'I_{coeff[0]}(D)')
        aux += coeff[1] * I_mi
      return I_bub
    
  def show(self):
    if self.reduced:
      I_bub = sp.symbols(f'I_{self.nu1,self.nu2}(D)')
      aux = 0
      for coeff in self.coefficients:
        I_mi = master_integrals_map[coeff[0]]
        aux += coeff[1] * I_mi
      return aux, I_bub
    else:
      print('The integral must be evaluated beforehand.')
      return

  def __str__(self):
      aux, I_bub = self.show()
      return_str = f'{I_bub} = {aux}'
      return return_str

if __name__ == "__main__":
  I_bub = BubbleIntegral(2,2)
  I_bub.reduce()
  print(I_bub)
