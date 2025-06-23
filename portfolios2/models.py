from django.db import models

class portfolios2(models.Model):
    logo = models.ImageField(upload_to='stock_logos/')  # Specifica la sottocartella
    name = models.CharField(max_length=100)
    ticker = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.name} ({self.ticker})"