# base image
FROM python:3.10

# working directory
WORKDIR /app

# copy files
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# run script
CMD ["python", "main.py"]
