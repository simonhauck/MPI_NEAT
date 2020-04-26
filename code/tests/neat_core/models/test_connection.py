from unittest import TestCase

from neat_core.models.connection import Connection


class ConnectionTest(TestCase):

    def test_connection(self):
        connection = Connection(10, 2, 3, 0.4, False)
        self.assertEqual(10, connection.innovation_number)

        self.assertEqual(2, connection.input_node)
        self.assertEqual(3, connection.output_node)
        self.assertEqual(0.4, connection.weight)
        self.assertFalse(connection.enabled)

        connection2 = Connection("test_number", 12, 13, 0.6, True, )
        self.assertEqual("test_number", connection2.innovation_number)
        self.assertEqual(12, connection2.input_node)
        self.assertEqual(13, connection2.output_node)
        self.assertEqual(0.6, connection2.weight)
        self.assertTrue(connection2.enabled)
