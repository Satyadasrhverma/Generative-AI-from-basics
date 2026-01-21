

from pydantic import BaseModel
from typing import Optional
class Student(BaseModel):
    name : str = 'nitish'
    age : Optional[int] = None
    
new_student = {'age' : 34}

student = Student(**new_student)
print(student)
