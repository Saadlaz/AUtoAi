# Generated by Django 5.1.3 on 2024-12-08 20:34

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0002_uploadeddataset"),
    ]

    operations = [
        migrations.DeleteModel(
            name="UploadedDataset",
        ),
    ]