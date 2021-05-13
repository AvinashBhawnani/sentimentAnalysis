from django.db import models

# Create your models here.
class Product(models.Model):
    asin=models.CharField(max_length=10, primary_key=True)
    title=models.TextField()
    price=models.TextField()
    def __str__(self):
        return(self.asin)

class Review(models.Model):
    asin=models.ForeignKey(Product,on_delete=models.CASCADE)
    text=models.TextField()
    rating=models.IntegerField()
    reviewer=models.TextField()
    def __str__(self):
        return(self.asin + self.reviewer)
