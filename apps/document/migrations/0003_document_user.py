# Generated by Django 2.2.2 on 2020-01-12 13:03

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('document', '0002_document_folder'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='user',
            field=models.ForeignKey(help_text='文章所属用户', on_delete=django.db.models.deletion.CASCADE, related_name='document', to=settings.AUTH_USER_MODEL, verbose_name='文章所属用户'),
        ),
    ]
