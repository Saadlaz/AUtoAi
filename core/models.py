from django.db import models

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')
    processed_file = models.FileField(upload_to='processed/', null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('uploaded', 'Uploaded'),
            ('preprocessed', 'Preprocessed'),
            ('visualized', 'Visualized'),
            ('classified', 'Classified'),
        ],
        default='uploaded',
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

