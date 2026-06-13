from http.server import BaseHTTPRequestHandler, HTTPServer
import requests

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            resp = requests.get("http://localhost:9002")
            data = resp.text
        except Exception as e:
            data = f"Error connecting to C++ server: {e}"

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        html = f"""
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Python Frontend</h1>
            <p>Data from C++ server:</p>
            <pre>{data}</pre>
        </body>
        </html>
        """

        self.wfile.write(html.encode("utf-8"))

def run(server_class=HTTPServer, handler_class=MyHandler, port=8080):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Serving at http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()