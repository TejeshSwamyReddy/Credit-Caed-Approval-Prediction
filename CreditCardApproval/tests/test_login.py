import unittest

from app import app


class LoginFlowTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_login_page_renders(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Email', response.data)
        self.assertIn(b'Password', response.data)

    def test_login_with_valid_credentials_redirects_to_home(self):
        response = self.client.post('/', data={
            'email': 'admin@apexbank.com',
            'password': 'admin123'
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/home')

    def test_login_with_invalid_credentials_stays_on_login_page(self):
        response = self.client.post('/', data={
            'email': 'wrong@example.com',
            'password': 'wrongpass'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid email or password', response.data)


if __name__ == '__main__':
    unittest.main()
