import "./globals.css";

export const metadata = {
  title: "Neural Nexus — AI Classroom Intelligence",
  description: "AI-powered classroom behavior analysis with real-time detection, tracking, and engagement metrics.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
