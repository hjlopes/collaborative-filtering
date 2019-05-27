# Collaborative filtering recommendation engine
Simple Collaborative recommendation engine model for product similarity estimation. 

There are two endpoints: 
- Item based collaborative filtering: the final model is based on item cosine distances;
- User based collaborative filtering: the final model is based on item euclidean distance.

Can be deployed using the Docker image.


## Requirements

### Running with Python only
Built with [Python 3.7](https://www.python.org/downloads/release/python-370/), but should work with other > Python 3.5 versions.

Install the requirements:

```bash
pip install -r requirements.txt
```

And run the in the _/app_ folder:
```bash
python main.py
```
### Running with a docker container
Requires that you have setup Docker and running. 

Simple run the command to build and deploy:

```docker
docker build -t recommender-image . 
docker run -p 5000:80 --name recommender recommender-image 
```

## Testing the application
There are two endpoints:
- **/predict-tag/** to obtain the TOP 10 similar tags, given another tag;
- **/predict-user/** to obtain the TOP 10 user recommended tags given a user id.

Do a GET petition to the local URL through browser with the user_id/tag_id as param:
```
http://localhost:5000/predict-tag/<tag_id>
http://localhost:5000/predict-user/<user_id>
```

Example:
http://localhost:5000/predict-tag/ff0d3fb21c00bc33f71187a2beec389e9eff5332

Or using curl:
```bash
curl http://lhttp://localhost:5000/predict-tag/ff0d3fb21c00bc33f71187a2beec389e9eff5332
```
You should see a response:
```json
{
    "predicted_tags":
    [
        "7ee223009403f7450993fe5d79516f1fc841e75e",
        "6b0cd6a8094daf42e766ea257a2af3571831bb32",
        "bdf147e99ee57500eb2dabcbf3cfa24e1daef357",
        "340f1eaf7ad0c07f1491338ab68cbcab30c315ec",
        "c093b1743115b3f9d368b2f7bdf54f367afccc7c",
        "61bc35a6401829bd28a8da47a2f235944ba8d2df",
        "85ef93bda0f7fb6327bd1b5ad44da26246b4360d",
        "dd3c8fd58366b577ce6b1d0f435602f11671c3dc",
        "551ec41539d9fb71200d18ec7903b1039cde594f",
        "ccc01cd0dd0becfcb86471efa1202f4a6c845545"
    ]
}
```
