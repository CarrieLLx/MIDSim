import uuid
import time


class MemoryItem:

    def __init__(
        self,
        agent_id,
        content,
        attributes=None,
        embedding=None,
        item_id=None,
        timestamp=None,
    ):
        self.agent_id = agent_id
        self.id = item_id if item_id is not None else uuid.uuid4()
        self.content = content
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.attributes = attributes if attributes is not None else {}
        self.embedding = embedding

    def __repr__(self):
        return f"MemoryItem(id={self.id}, content='{self.content}', attributes={self.attributes})"

    def to_dict(self):
        """
        将MemoryItem对象转换为可JSON序列化的字典。
        """
        embedding_serializable = self.embedding
        if hasattr(self.embedding, 'tolist'):
            embedding_serializable = self.embedding.tolist()

        return {
            'agent_id': self.agent_id,
            'id': str(self.id),
            'content': self.content,
            'timestamp': self.timestamp,
            'attributes': self.attributes,
            'embedding': embedding_serializable,
        }

    @classmethod
    def from_dict(cls, data):
        """
        从字典创建MemoryItem对象。
        """
        embedding = data.get('embedding')
        if isinstance(embedding, list):
            try:
                import numpy as np

                embedding = np.array(embedding)
            except ImportError:
                pass

        return cls(
            agent_id=data['agent_id'],
            content=data['content'],
            attributes=data.get('attributes'),
            embedding=embedding,
            item_id=uuid.UUID(data['id']),
            timestamp=data['timestamp'],
        )
