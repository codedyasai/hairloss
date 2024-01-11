from django.db import models

# Create your models here.
class Human(models.Model):
    id = models.AutoField(primary_key=True, max_length=100)
    sex = models.CharField(max_length=40)
    age = models.CharField(max_length=40)
    pred = models.CharField(max_length=40)
    skin = models.CharField(max_length=40, default='default_value')
    def __str__(self):
        return f'{self.age} , {self.sex} , {self.pred}, {self.skin}'

