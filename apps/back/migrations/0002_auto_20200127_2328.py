# Generated by Django 2.2.2 on 2020-01-27 23:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('back', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='admincountmodel',
            name='login_count',
        ),
        migrations.AlterField(
            model_name='admincountmodel',
            name='entitynaming_count',
            field=models.IntegerField(default=0, help_text='实体命名', verbose_name='实体命名'),
        ),
        migrations.AlterField(
            model_name='admincountmodel',
            name='knowpic_count',
            field=models.IntegerField(default=0, help_text='使用识别次数', verbose_name='使用识别次数'),
        ),
        migrations.AlterField(
            model_name='admincountmodel',
            name='read_count',
            field=models.IntegerField(default=0, help_text='语音功能', verbose_name='语音功能'),
        ),
        migrations.AlterField(
            model_name='admincountmodel',
            name='translate_count',
            field=models.IntegerField(default=0, help_text='翻译功能', verbose_name='翻译功能'),
        ),
        migrations.AlterField(
            model_name='admincountmodel',
            name='upload_count',
            field=models.IntegerField(default=0, help_text='上传次数', verbose_name='上传次数'),
        ),
    ]
