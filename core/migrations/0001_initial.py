# Generated by Django 5.1.3 on 2024-12-06 16:56

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Dataset",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                ("file", models.FileField(upload_to="datasets/")),
                (
                    "processed_file",
                    models.FileField(blank=True, null=True, upload_to="processed/"),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("uploaded", "Uploaded"),
                            ("preprocessed", "Preprocessed"),
                            ("visualized", "Visualized"),
                            ("classified", "Classified"),
                        ],
                        default="uploaded",
                        max_length=20,
                    ),
                ),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]