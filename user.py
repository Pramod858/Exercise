from zenml.client import Client

client = Client()
client.create_secret(
    name="my_secret",
    values={
        "username": "admin",
        "password": "abc123"
    }
)