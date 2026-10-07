import { useEffect, useState } from "react";

export default function App() {
  const [digest, setDigest] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(null);

  useEffect(() => {
    fetch("/digest/today")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status === 404 ? "No digest yet." : `Error ${r.status}`)))
      .then((d) => {
        setDigest(d);
        // Opening the page is what counts as "seen".
        fetch(`/digest/${d.date}/opened`, { method: "PUT" });
      })
      .catch((e) => setError(String(e)));
  }, []);

  async function toggle(item) {
    setBusy(item.id);
    const r = await fetch(`/papers/${item.id}/not-interested`, { method: item.rejected ? "DELETE" : "PUT" })
      .catch(() => ({ ok: false, status: "network" }))
      .finally(() => setBusy(null));
    if (!r.ok) return setError(`Could not save that (error ${r.status}).`);
    setDigest((d) => ({
      ...d,
      items: d.items.map((i) => (i.id === item.id ? { ...i, rejected: !i.rejected } : i)),
    }));
  }

  if (error) return <main><p className="banner">{error}</p></main>;
  if (!digest) return <main><p>Loading…</p></main>;

  return (
    <main>
      <h1>arXiv papers for {digest.date}</h1>
      {digest.stale && <p className="banner">Today's digest is not ready. This is the last one.</p>}
      {digest.items.map((item) => (
        <article key={item.id} className={item.rejected ? "rejected" : ""}>
          <h2><a href={item.link} target="_blank" rel="noreferrer">{item.headline}</a></h2>
          <p className="paper-title">{item.title}</p>
          <ul>{item.bullets.map((b, n) => <li key={n}>{b}</li>)}</ul>
          {item.reason && <p className="reason">{item.reason}</p>}
          <button disabled={busy === item.id} onClick={() => toggle(item)}>{item.rejected ? "Undo" : "Not interested"}</button>
        </article>
      ))}
    </main>
  );
}
