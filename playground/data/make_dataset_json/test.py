

dataset_content = """
[
    {
        "id": "test-628d-4724-b370-b84de974a19f",
        "image": "test/1.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWho is in the picture?"
            },
            {
                "from": "gpt",
                "value": "The person in the picture is Nathida, who is a character in the Original God and its derivative works produced by Mihoyo. Her real name is Buyel, the grass god in the \"Earthly Seven rulers\", and is given the nickname of \"Little Lucky Grass King\" by the XuMi people, the youngest of the seven gods today. "
            }
        ]
    }
]
"""

with open("test.json", "w") as f:
    f.write(dataset_content)
