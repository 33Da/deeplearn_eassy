# Generated by Django 2.2.2 on 2020-01-12 13:03

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('folder', '0001_initial'),
        ('document', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='folder',
            field=models.ForeignKey(blank=True, default=1, help_text='所属文件夹', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='document', to='folder.Folder', verbose_name='所属文件夹'),
        ),
    ]
