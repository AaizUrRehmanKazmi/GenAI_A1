export async function getHealth() {
  const response = await fetch('/api/health');
  if (!response.ok) throw new Error(`Health check failed (${response.status})`);
  return response.json();
}
