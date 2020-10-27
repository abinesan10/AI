# Generated by Django 3.0.7 on 2020-10-27 20:32

from django.db import migrations, models
import jsonfield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='photoGallery',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dates', models.DateTimeField(auto_now_add=True)),
                ('profile_picture', jsonfield.fields.JSONField(null=True)),
                ('image_type', models.CharField(max_length=80)),
            ],
        ),
        migrations.RemoveField(
            model_name='registerinfo',
            name='mobileVerifiedStatus',
        ),
        migrations.AddField(
            model_name='registerinfo',
            name='mobileCountryCode',
            field=models.CharField(max_length=30, null=True),
        ),
    ]