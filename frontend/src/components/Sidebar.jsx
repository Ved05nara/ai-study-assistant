import { useState, useRef, useCallback } from "react";
import { uploadMultipleFiles } from "../services/api";
function formatDate(iso) {
    if (!iso) return "";
    try {
        return new Date(iso).toLocaleDateString(undefined, {
            month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        });
    } catch {
        return "";
    }
}

export default function Sidebar({ documents, onDocumentsChange, onToast }) {
    const [dragOver, setDragOver] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [confirmDelete, setConfirmDelete] = useState(null); // filename to confirm
    const fileInputRef = useRef(null);

    const handleFiles = useCallback(async (files) => {
        const pdfs = Array.from(files).filter((f) => f.name.endsWith(".pdf"));
        if (!pdfs.length) {
            onToast("Only PDF files are supported.", "error");
            return;
        }
        setUploading(true);
        setProgress(0);
        try {
            const res = await uploadMultipleFiles(pdfs, (e) => {
                if (e.total) setProgress(Math.round((e.loaded / e.total) * 100));
            });
            const data = res.data;
            const ok = data.results?.filter((r) => !r.error).length || 0;
            const errs = data.results?.filter((r) => r.error).length || 0;
            if (ok > 0) onToast(`${ok} file${ok > 1 ? "s" : ""} uploaded successfully! 🎉`, "success");
            if (errs > 0) onToast(`${errs} file${errs > 1 ? "s" : ""} failed to upload.`, "error");
            onDocumentsChange();
        } catch (err) {
            onToast("Upload failed. Please try again.", "error");
        } finally {
            setUploading(false);
            setProgress(0);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    }, [onDocumentsChange, onToast]);

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
    };

    const handleDeleteConfirm = async () => {
        const filename = confirmDelete;
        setConfirmDelete(null);
        try {
            const { deleteDocument } = await import("../services/api");
            await deleteDocument(filename);
            onToast(`"${filename}" deleted.`, "info");
            onDocumentsChange();
        } catch {
            onToast("Failed to delete document.", "error");
        }
    };

    return (
        <>
            <aside className="sidebar">
                {/* Logo */}
                <div className="sidebar-header">
                    <div className="sidebar-logo">
                        <div className="sidebar-logo-icon"><img src="campusGPT_Logo2.png" alt="Logo" /></div>
                        <div className="sidebar-logo-text">
                            <h1>CampusGPT</h1>
                            <span>AI-Powered Notes</span>
                        </div>
                    </div>

                    {/* Upload Drop Zone */}
                    <div
                        className={`upload-zone${dragOver ? " drag-over" : ""}`}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".pdf"
                            multiple
                            onChange={(e) => handleFiles(e.target.files)}
                            disabled={uploading}
                        />
                        <div className="upload-icon">
                            {uploading ? "⏳" : "☁️"}
                        </div>
                        <p>
                            {uploading
                                ? `Uploading… ${progress}%`
                                : <>Drop <strong>PDF files</strong> here<br />or click to browse</>
                            }
                        </p>
                        {uploading && (
                            <div className="upload-progress">
                                <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
                            </div>
                        )}
                    </div>
                </div>

                {/* Document List */}
                <div className="sidebar-docs">
                    <p className="sidebar-section-label">
                        Documents ({documents.length})
                    </p>
                    {documents.length === 0 ? (
                        <div className="no-docs">
                            <div style={{ fontSize: 28, marginBottom: 8 }}>📄</div>
                            <p>No documents yet.<br />Upload a PDF to get started.</p>
                        </div>
                    ) : (
                        documents.map((doc) => (
                            <div key={doc.filename} className="doc-item">
                                <span className="doc-icon">📄</span>
                                <div className="doc-info">
                                    <div className="doc-name" title={doc.filename}>
                                        {doc.filename}
                                    </div>
                                    <div className="doc-meta">
                                        <span className="doc-badge">{doc.chunk_count} chunks</span>
                                        <span className="doc-time">{formatDate(doc.upload_time)}</span>
                                    </div>
                                </div>
                                <button
                                    className="doc-delete-btn"
                                    title={`Delete ${doc.filename}`}
                                    onClick={() => setConfirmDelete(doc.filename)}
                                >
                                    🗑️
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </aside>

            {/* Confirm Delete Dialog */}
            {confirmDelete && (
                <div className="dialog-overlay" onClick={() => setConfirmDelete(null)}>
                    <div className="dialog" onClick={(e) => e.stopPropagation()}>
                        <h3>Delete Document?</h3>
                        <p>
                            This will permanently remove <strong>"{confirmDelete}"</strong> from the
                            index. You'll need to re-upload it to use it again.
                        </p>
                        <div className="dialog-actions">
                            <button className="btn btn-ghost" onClick={() => setConfirmDelete(null)}>
                                Cancel
                            </button>
                            <button className="btn btn-danger" onClick={handleDeleteConfirm}>
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
