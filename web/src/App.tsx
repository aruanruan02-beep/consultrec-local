import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  Checkbox,
  DatePicker,
  Descriptions,
  Drawer,
  Empty,
  Form,
  Grid,
  Input,
  Layout,
  Menu,
  Message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Spin,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
} from "@arco-design/web-react";
import {
  IconArrowLeft,
  IconEdit,
  IconFileAudio,
  IconHome,
  IconRefresh,
  IconRobot,
  IconSettings,
  IconUpload,
  IconFolder,
} from "@arco-design/web-react/icon";

const { Sider, Content } = Layout;
const { Title, Text } = Typography;
const { Row, Col } = Grid;
const FormItem = Form.Item;
const TextArea = Input.TextArea;

const PROCESSING_STATUSES = new Set([
  "uploaded",
  "queued_transcription",
  "transcribing",
  "diarizing",
  "queued_note_generation",
  "generating_note",
]);

const DEFAULT_SETTINGS: Settings = {
  data_root: "data",
  whisper_command:
    ".venv/bin/python scripts/transcribe_whisperx.py --audio {audio} --output {transcript_json} --model small --language zh --device cpu --compute-type int8 --min-speakers 2 --max-speakers 2",
  diarization_command: "",
  llm_command: ".venv/bin/python scripts/generate_note_ollama.py --prompt-file {prompt_file} --model qwen2.5:7b-instruct",
  clinical_prompt_path: "prompts/clinical_note_prompt.md",
  hf_token: "",
  prompt_template: "",
};

type ViewName = "records" | "session" | "settings";
type Speaker = "咨询师" | "来访者" | "说话人 1" | "说话人 2" | "未确认";

type CaseItem = {
  case_id: string;
  note?: string;
};

type SessionSummary = {
  case_id: string;
  session_id: string;
  session_date?: string;
  file_name?: string;
  status?: string;
  progress_label?: string;
  error_message?: string;
  is_processing?: boolean;
  next_action?: string;
};

type TranscriptSegment = {
  start: number;
  end: number;
  speaker?: Speaker | string;
  text?: string;
  [key: string]: unknown;
};

type ClinicalNote = {
  session_summary?: Record<string, string[]>;
  soap?: Record<string, string[]>;
  raw_text?: string;
};

type SessionDetail = SessionSummary & {
  diarization_configured?: boolean;
  transcript?: TranscriptSegment[];
  clinical_note?: ClinicalNote;
  audio_path?: string;
  transcript_path?: string;
  note_json_path?: string;
  markdown_path?: string;
};

type Settings = {
  data_root: string;
  whisper_command: string;
  diarization_command: string;
  llm_command: string;
  clinical_prompt_path: string;
  hf_token: string;
  prompt_template?: string;
};

type UploadValues = {
  case_id: string;
  session_date: string;
  note?: string;
  audio?: Array<{ originFile?: File }>;
};

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    let msg = response.statusText;
    try {
      const data = await response.json();
      if (data && data.detail) {
        if (typeof data.detail === "string") {
          msg = data.detail;
        } else if (Array.isArray(data.detail)) {
          msg = data.detail.map((err: any) => `${err.loc?.join(".") || "field"}: ${err.msg}`).join("; ");
        } else if (typeof data.detail === "object") {
          msg = JSON.stringify(data.detail);
        }
      }
    } catch (e) {
      try {
        const text = await response.text();
        if (text) msg = text;
      } catch (inner) {}
    }
    throw new Error(msg);
  }
  return response.json();
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

function statusColor(status?: string) {
  if (status === "complete") return "green";
  if (status === "error") return "red";
  if (status === "canceled") return "gray";
  if (status === "awaiting_review") return "orange";
  if (status && PROCESSING_STATUSES.has(status)) return "arcoblue";
  return "gray";
}

function progressHint(item: Partial<SessionSummary>) {
  if (item.error_message) return "处理失败，可点击重试或进入详情查看";
  if (item.is_processing) return "系统正在本地处理，请稍候";
  if (item.status === "awaiting_review") return "请确认咨询师 / 来访者角色";
  if (item.status === "complete") return "SOAP 与 Summary 已生成";
  return "等待处理";
}

function formatTime(value: number) {
  const total = Math.max(0, Math.round(Number(value) || 0));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function formatDateLabel(value?: string) {
  if (!value) return "未设置日期";
  const parts = String(value).split("-");
  if (parts.length !== 3) return value;
  const [year, month, day] = parts;
  return `${year}年${Number(month)}月${Number(day)}日`;
}

function normalizeSpeaker(value?: string): Speaker {
  if (value === "Therapist") return "咨询师";
  if (value === "Client") return "来访者";
  if (value === "咨询师" || value === "来访者" || value === "说话人 1" || value === "说话人 2") return value;
  return "未确认";
}

function summarizeWhisper(command: string) {
  if (!command) return "尚未配置";
  const modelMatch = command.match(/--model\s+([^\s]+)/);
  if (command.includes("transcribe_whisperx.py") || command.includes("whisperx")) {
    return `WhisperX ${modelMatch ? modelMatch[1] : "本地模型"}`;
  }
  return `Whisper ${modelMatch ? modelMatch[1] : "本地模型"}`;
}

function summarizeLLM(command: string) {
  if (!command) return "尚未配置";
  const modelMatch = command.match(/--model\s+([^\s]+)/);
  return modelMatch ? `Ollama ${modelMatch[1]}` : "Ollama 本地模型";
}

function summarizeDiarization(command: string) {
  if (!command) return "已并入 WhisperX 转写";
  if (command.includes("diarize_voice_features.py")) return "本地声音特征分组";
  return "已配置";
}

function StatusTag({ status, label }: { status?: string; label?: string }) {
  return <Tag color={statusColor(status)}>{label || status || "未知状态"}</Tag>;
}

function NoteList({ items }: { items?: string[] }) {
  const values = items && items.length ? items : ["逐字稿中未明确提及"];
  return (
    <ul className="note-list">
      {values.map((item, index) => (
        <li key={`${item}-${index}`}>{item}</li>
      ))}
    </ul>
  );
}

export default function App() {
  const [view, setView] = useState<ViewName>("records");
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSession, setCurrentSession] = useState<SessionDetail | null>(null);
  const [caseQuery, setCaseQuery] = useState("");
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [uploadVisible, setUploadVisible] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [savingSettings, setSavingSettings] = useState(false);
  const [transcriptDraft, setTranscriptDraft] = useState<TranscriptSegment[]>([]);
  const [uploadForm] = Form.useForm<UploadValues>();
  const [repoForm] = Form.useForm();
  const [hfForm] = Form.useForm();
  const [commandsForm] = Form.useForm();
  const [promptText, setPromptText] = useState("");
  const [tempPromptText, setTempPromptText] = useState("");
  const [promptDrawerVisible, setPromptDrawerVisible] = useState(false);
  const [isRepoDirty, setIsRepoDirty] = useState(false);
  const [isHfDirty, setIsHfDirty] = useState(false);
  const [isCommandsDirty, setIsCommandsDirty] = useState(false);
  const pollTimer = useRef<number | null>(null);

  const loadCases = useCallback(async () => {
    const data = await api<{ cases: CaseItem[] }>("/api/cases");
    setCases(data.cases || []);
  }, []);

  const loadSessions = useCallback(async () => {
    const query = new URLSearchParams();
    if (caseQuery.trim()) query.set("case", caseQuery.trim());
    setLoadingSessions(true);
    try {
      const data = await api<{ sessions: SessionSummary[] }>(`/api/sessions?${query.toString()}`);
      setSessions(data.sessions || []);
    } finally {
      setLoadingSessions(false);
    }
  }, [caseQuery]);

  const openSession = useCallback(async (caseId: string, sessionId: string) => {
    const data = await api<SessionDetail>(
      `/api/sessions/${encodeURIComponent(caseId)}/${encodeURIComponent(sessionId)}`
    );
    setCurrentSession(data);
    setTranscriptDraft(data.transcript || []);
    setView("session");
  }, []);

  const refreshCurrentSession = useCallback(async () => {
    if (!currentSession) return;
    const data = await api<SessionDetail>(
      `/api/sessions/${encodeURIComponent(currentSession.case_id)}/${encodeURIComponent(currentSession.session_id)}`
    );
    setCurrentSession(data);
    setTranscriptDraft(data.transcript || []);
  }, [currentSession]);

  const loadSettings = useCallback(async () => {
    try {
      const data = { ...DEFAULT_SETTINGS, ...(await api<Partial<Settings>>("/api/config")) };
      setSettings(data);
      repoForm.setFieldsValue({
        data_root: data.data_root,
        clinical_prompt_path: data.clinical_prompt_path,
      });
      hfForm.setFieldsValue({
        hf_token: data.hf_token,
      });
      commandsForm.setFieldsValue({
        whisper_command: data.whisper_command,
        llm_command: data.llm_command,
      });
      setPromptText(data.prompt_template || "");
      setTempPromptText(data.prompt_template || "");
    } catch (error) {
      console.warn("Unable to load settings, showing defaults.", error);
      repoForm.setFieldsValue(DEFAULT_SETTINGS);
      hfForm.setFieldsValue(DEFAULT_SETTINGS);
      commandsForm.setFieldsValue(DEFAULT_SETTINGS);
      setPromptText("");
      setTempPromptText("");
    }
  }, [repoForm, hfForm, commandsForm]);

  useEffect(() => {
    loadCases().catch((error) => console.warn("Unable to load cases.", error));
    loadSettings();
  }, [loadCases, loadSettings]);

  useEffect(() => {
    loadSessions().catch((error) => {
      Message.error(`加载记录失败：${error.message}`);
    });
  }, [loadSessions]);

  useEffect(() => {
    if (pollTimer.current) window.clearTimeout(pollTimer.current);
    const shouldPoll =
      sessions.some((item) => item.is_processing) ||
      Boolean(currentSession && PROCESSING_STATUSES.has(currentSession.status || ""));
    if (shouldPoll) {
      pollTimer.current = window.setTimeout(async () => {
        await loadSessions();
        if (currentSession) await refreshCurrentSession();
      }, 2500);
    }
    return () => {
      if (pollTimer.current) window.clearTimeout(pollTimer.current);
    };
  }, [sessions, currentSession, loadSessions, refreshCurrentSession]);

  const retryTranscription = async (caseId: string, sessionId: string) => {
    await api(`/api/sessions/${caseId}/${sessionId}/transcribe`, { method: "POST" });
    Message.success("已重新加入转写队列");
    await loadSessions();
    if (currentSession?.session_id === sessionId) await refreshCurrentSession();
  };

  const rerunDiarization = async (caseId: string, sessionId: string) => {
    await api(`/api/sessions/${caseId}/${sessionId}/diarize`, { method: "POST" });
    Message.success("已开始重新自动整理");
    await loadSessions();
    await refreshCurrentSession();
  };

  const deleteSession = async (item: SessionSummary) => {
    await api(`/api/sessions/${item.case_id}/${item.session_id}`, { method: "DELETE" });
    Message.success("记录已删除");
    if (currentSession?.session_id === item.session_id) {
      setCurrentSession(null);
      setView("records");
    }
    await loadCases();
    await loadSessions();
  };

  const cancelSession = async (item: SessionSummary) => {
    await api(`/api/sessions/${item.case_id}/${item.session_id}/cancel`, { method: "POST" });
    Message.success("已取消当前处理任务");
    await loadSessions();
    if (currentSession?.session_id === item.session_id) await refreshCurrentSession();
  };

  const saveTranscript = async (data: SessionDetail) => {
    await api(`/api/sessions/${data.case_id}/${data.session_id}/transcript`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segments: transcriptDraft }),
    });
    Message.success("逐字稿修改已保存");
  };

  const saveClinicalNote = async (data: SessionDetail, notePayload: ClinicalNote) => {
    const updated = await api<SessionDetail>(`/api/sessions/${data.case_id}/${data.session_id}/note`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(notePayload),
    });
    Message.success("临床记录修改已保存");
    setCurrentSession(updated);
    await loadSessions();
  };

  const submitReview = async (data: SessionDetail) => {
    const run = async () => {
      await api(`/api/sessions/${data.case_id}/${data.session_id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments: transcriptDraft }),
      });
      Message.success("已开始生成临床记录");
      await refreshCurrentSession();
      await loadSessions();
    };

    if (transcriptDraft.some((item) => !["咨询师", "来访者"].includes(normalizeSpeaker(String(item.speaker))))) {
      Modal.confirm({
        title: "还有未确认角色",
        content: "仍有片段保留为说话人标签或未确认，仍然继续生成记录吗？",
        okText: "继续生成",
        cancelText: "返回检查",
        onOk: run,
      });
      return;
    }
    await run();
  };

  const submitUpload = async () => {
    const values = await uploadForm.validate();
    const file = values.audio?.[0]?.originFile;
    if (!file) {
      Message.warning("请选择录音文件");
      return;
    }
    const formData = new FormData();
    formData.append("case_id", values.case_id);
    formData.append("session_date", values.session_date || today());
    formData.append("note", values.note || "");
    formData.append("audio", file);
    setUploading(true);
    try {
      const session = await api<SessionSummary>("/api/sessions", { method: "POST", body: formData });
      Message.success("录音已上传，开始处理");
      setUploadVisible(false);
      uploadForm.resetFields();
      uploadForm.setFieldsValue({ session_date: today() });
      await loadCases();
      await loadSessions();
      await openSession(session.case_id, session.session_id);
    } finally {
      setUploading(false);
    }
  };

  const saveConfig = async (patch: Partial<Settings>) => {
    try {
      const currentValues = {
        data_root: repoForm.getFieldValue("data_root") ?? settings.data_root,
        clinical_prompt_path: repoForm.getFieldValue("clinical_prompt_path") ?? settings.clinical_prompt_path,
        hf_token: hfForm.getFieldValue("hf_token") ?? settings.hf_token,
        whisper_command: commandsForm.getFieldValue("whisper_command") ?? settings.whisper_command,
        llm_command: commandsForm.getFieldValue("llm_command") ?? settings.llm_command,
        prompt_template: promptText,
        ...patch,
      };
      const data = await api<Settings>("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentValues),
      });
      setSettings(data);
      repoForm.setFieldsValue({
        data_root: data.data_root,
        clinical_prompt_path: data.clinical_prompt_path,
      });
      hfForm.setFieldsValue({
        hf_token: data.hf_token,
      });
      commandsForm.setFieldsValue({
        whisper_command: data.whisper_command,
        llm_command: data.llm_command,
      });
      setPromptText(data.prompt_template || "");
      setTempPromptText(data.prompt_template || "");
      Message.success("设置已保存");
    } catch (error: any) {
      Message.error("保存设置失败: " + (error instanceof Error ? error.message : String(error)));
      console.error(error);
    }
  };

  const saveRepoSettings = async () => {
    const values = await repoForm.validate();
    await saveConfig(values);
  };

  const handleSelectDirectory = async () => {
    try {
      const res = await api<{ path?: string; error?: string }>("/api/config/select_directory", {
        method: "POST",
      });
      if (res.path) {
        repoForm.setFieldValue("data_root", res.path);
        setIsRepoDirty(true);
      } else if (res.error && res.error !== "User canceled.") {
        Message.error(`选择文件夹失败: ${res.error}`);
      }
    } catch (error: any) {
      Message.error("选择文件夹时发生错误: " + (error instanceof Error ? error.message : String(error)));
      console.error(error);
    }
  };

  const saveHfSettings = async () => {
    const values = await hfForm.validate();
    await saveConfig(values);
  };

  const saveCommandsSettings = async () => {
    const values = await commandsForm.validate();
    await saveConfig(values);
  };

  const savePromptTemplate = async (newText: string) => {
    await saveConfig({ prompt_template: newText });
  };

  const columns = useMemo(
    () => [
      {
        title: "文件",
        dataIndex: "file_name",
        render: (_: unknown, item: SessionSummary) => (
          <Space>
            <span className="file-mark">
              <IconFileAudio />
            </span>
            <div className="record-copy">
              <Text className="record-name">{item.file_name || "录音文件"}</Text>
              <Text type="secondary" className="record-id">
                {item.session_id}
              </Text>
            </div>
          </Space>
        ),
      },
      { title: "Case", dataIndex: "case_id" },
      {
        title: "日期",
        dataIndex: "session_date",
        render: (value: string) => value || "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        align: "center" as const,
        render: (_: unknown, item: SessionSummary) => <StatusTag status={item.status} label={item.progress_label} />,
      },
      {
        title: "操作",
        dataIndex: "actions",
        align: "left" as const,
        render: (_: unknown, item: SessionSummary) => (
          <Space size={14} align="center">
            <Button className="record-action-button" type="text" size="small" onClick={() => openSession(item.case_id, item.session_id)}>
              查看详情
            </Button>
            {item.is_processing && (
              <Popconfirm
                title="取消只会停止当前本地处理，不会删除已经生成的文件。"
                okText="取消任务"
                cancelText="返回"
                onOk={() => cancelSession(item)}
              >
                <Button className="record-action-button" type="text" status="warning" size="small">
                  取消任务
                </Button>
              </Popconfirm>
            )}
            {item.next_action === "retry" && (
              <Button className="record-action-button" type="text" size="small" onClick={() => retryTranscription(item.case_id, item.session_id)}>
                重试
              </Button>
            )}
            <Popconfirm
              title={`确定删除“${item.file_name || "这条录音"}”吗？删除后不能恢复。`}
              okText="删除"
              cancelText="取消"
              onOk={() => deleteSession(item)}
            >
              <Button className="record-action-button record-action-danger" type="text" size="small">
                删除
              </Button>
            </Popconfirm>
          </Space>
        ),
      },
    ],
    [openSession, currentSession, transcriptDraft]
  );

  return (
    <Layout className="app-shell">
      <Sider className="app-sider" width={224}>
        <div className="brand">
          <span className="brand-mark">M</span>
          <span>咨询记录</span>
        </div>
        <Menu selectedKeys={[view === "session" ? "records" : view]} onClickMenuItem={(key) => setView(key as ViewName)}>
          <Menu.Item key="records">
            <IconHome />
            记录
          </Menu.Item>
          <Menu.Item key="settings">
            <IconSettings />
            设置
          </Menu.Item>
        </Menu>
      </Sider>

      <Content className="app-content">
        {view === "records" && (
          <section className="page">
            <div className="page-header">
              <Title heading={4}>录音记录</Title>
              <Space>
                <Input.Search
                  allowClear
                  placeholder="搜索 Case ID"
                  value={caseQuery}
                  onChange={setCaseQuery}
                  className="case-search"
                />
                <Button
                  type="primary"
                  icon={<IconUpload />}
                  onClick={() => {
                    uploadForm.setFieldsValue({ session_date: today() });
                    setUploadVisible(true);
                  }}
                >
                  上传录音
                </Button>
              </Space>
            </div>

            <Table
              rowKey="session_id"
              loading={loadingSessions}
              columns={columns}
              data={sessions}
              pagination={false}
              noDataElement={<Empty description="还没有记录。点击右上角“上传录音”开始处理。" />}
            />
          </section>
        )}

        {view === "session" && currentSession && (
          <SessionDetailView
            data={currentSession}
            transcriptDraft={transcriptDraft}
            setTranscriptDraft={setTranscriptDraft}
            onBack={async () => {
              setCurrentSession(null);
              setView("records");
              await loadSessions();
            }}
            onRefresh={refreshCurrentSession}
            onRetry={() => retryTranscription(currentSession.case_id, currentSession.session_id)}
            onRerunDiarization={() => rerunDiarization(currentSession.case_id, currentSession.session_id)}
            onRegenerate={() => submitReview(currentSession)}
            onSaveTranscript={async () => {
              await saveTranscript(currentSession);
              await refreshCurrentSession();
              await loadSessions();
            }}
            onSubmitReview={() => submitReview(currentSession)}
            onSaveClinicalNote={async (notePayload) => {
              if (currentSession) {
                await saveClinicalNote(currentSession, notePayload);
              }
            }}
          />
        )}

        {view === "settings" && (
          <section className="page settings-page">
            <div className="page-header settings-header">
              <Title heading={4}>设置</Title>
            </div>

            {/* Block 1: 提示词 (放在最上面) */}
            <div className="settings-block">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <Title heading={6} style={{ margin: 0 }}>提示词</Title>
                <Button type="primary" icon={<IconEdit />} onClick={() => {
                  setTempPromptText(promptText);
                  setPromptDrawerVisible(true);
                }}>
                  编辑提示词
                </Button>
              </div>
              <div style={{ marginBottom: 12 }}>
                <Text type="secondary">
                  自定义大模型提取咨询记录 (Session Summary & SOAP) 的模板和生成规范。
                </Text>
              </div>
              <div style={{
                padding: "12px 16px",
                background: "var(--color-fill-1, #f2f3f5)",
                borderRadius: 4,
                maxHeight: 140,
                overflowY: "auto",
                fontFamily: "monospace",
                fontSize: 12,
                color: "var(--color-text-2, #4e5969)",
                border: "1px solid var(--color-border-1, #f2f3f5)",
                whiteSpace: "pre-wrap"
              }}>
                {promptText || "暂无提示词模板"}
              </div>
            </div>

            {/* Block 2: 数据保存位置 (单独列出, 用选择文件夹方式) */}
            <div className="settings-block">
              <Title heading={6}>数据保存位置</Title>
              <Form form={repoForm} layout="vertical" onValuesChange={() => setIsRepoDirty(true)}>
                <FormItem label="本地保存位置" field="data_root" rules={[{ required: true, message: "请选择或输入数据保存位置" }]}>
                  <Input
                    placeholder="请选择或输入保存数据的绝对路径"
                    addAfter={
                      <Button
                        type="text"
                        size="small"
                        icon={<IconFolder />}
                        onClick={handleSelectDirectory}
                        style={{ padding: "0 8px", height: "auto" }}
                      >
                        选择文件夹
                      </Button>
                    }
                  />
                </FormItem>
                {isRepoDirty && (
                  <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 8 }}>
                    <Space>
                      <Button size="small" onClick={() => {
                        repoForm.setFieldsValue({
                          data_root: settings.data_root,
                        });
                        setIsRepoDirty(false);
                      }}>
                        取消
                      </Button>
                      <Button type="primary" size="small" onClick={async () => {
                        await saveRepoSettings();
                        setIsRepoDirty(false);
                      }}>
                        保存
                      </Button>
                    </Space>
                  </div>
                )}
              </Form>
            </div>

            {/* Block 3: Hugging Face 设置 */}
            <div className="settings-block">
              <Title heading={6}>Hugging Face 设置</Title>
              <Form form={hfForm} layout="vertical" onValuesChange={() => setIsHfDirty(true)}>
                <FormItem label="Hugging Face Read Token (HF_TOKEN) - 用于 WhisperX 说话人区分功能" field="hf_token">
                  <Input.Password placeholder="hf_..." />
                </FormItem>
                {isHfDirty && (
                  <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 8 }}>
                    <Space>
                      <Button size="small" onClick={() => {
                        hfForm.setFieldsValue({
                          hf_token: settings.hf_token,
                        });
                        setIsHfDirty(false);
                      }}>
                        取消
                      </Button>
                      <Button type="primary" size="small" onClick={async () => {
                        await saveHfSettings();
                        setIsHfDirty(false);
                      }}>
                        保存
                      </Button>
                    </Space>
                  </div>
                )}
              </Form>
            </div>
          </section>
        )}
      </Content>

      <Modal
        title="上传录音"
        visible={uploadVisible}
        onCancel={() => setUploadVisible(false)}
        onOk={submitUpload}
        confirmLoading={uploading}
        okText="上传并处理"
        cancelText="取消"
        unmountOnExit
      >
        <Form form={uploadForm} layout="vertical" initialValues={{ session_date: today() }}>
          <FormItem label="Case ID" field="case_id" rules={[{ required: true, message: "请输入 Case ID" }]}>
            <Input placeholder="例如 C-001" list="caseOptions" />
          </FormItem>
          <datalist id="caseOptions">
            {cases.map((item) => (
              <option key={item.case_id} value={item.case_id} />
            ))}
          </datalist>
          <FormItem label="会谈日期" field="session_date" rules={[{ required: true, message: "请选择会谈日期" }]}>
            <DatePicker className="full-width" onChange={(dateString) => uploadForm.setFieldValue("session_date", dateString)} />
          </FormItem>
          <FormItem label="Case 备注" field="note">
            <TextArea autoSize={{ minRows: 2, maxRows: 4 }} placeholder="可选" />
          </FormItem>
          <FormItem label="录音文件" field="audio" rules={[{ required: true, message: "请选择录音文件" }]}>
            <Upload
              drag
              limit={1}
              accept=".mp3,.wav,.m4a,audio/*"
              autoUpload={false}
              tip="支持 mp3 / wav / m4a。上传后会自动开始转写。"
            />
          </FormItem>
        </Form>
      </Modal>

      <Drawer
        title="编辑提示词模板"
        visible={promptDrawerVisible}
        width={640}
        onCancel={() => {
          setPromptDrawerVisible(false);
          setTempPromptText(promptText);
        }}
        onOk={async () => {
          await savePromptTemplate(tempPromptText);
          setPromptDrawerVisible(false);
        }}
        okText="保存"
        cancelText="取消"
        unmountOnExit
      >
        <div style={{ marginBottom: 16 }}>
          <Text type="secondary">
            提示词模板中必须包含 <code>{"{{TRANSCRIPT}}"}</code> 占位符，系统会在生成时自动将其替换为咨询逐字稿。
          </Text>
        </div>
        <TextArea
          value={tempPromptText}
          onChange={(val) => setTempPromptText(val)}
          style={{ height: "calc(100vh - 220px)", fontFamily: "monospace", fontSize: 13, lineHeight: 1.5 }}
          placeholder="输入大模型的提示词规范..."
        />
      </Drawer>
    </Layout>
  );
}

function SessionDetailView({
  data,
  transcriptDraft,
  setTranscriptDraft,
  onBack,
  onRefresh,
  onRetry,
  onRerunDiarization,
  onRegenerate,
  onSaveTranscript,
  onSubmitReview,
  onSaveClinicalNote,
}: {
  data: SessionDetail;
  transcriptDraft: TranscriptSegment[];
  setTranscriptDraft: (segments: TranscriptSegment[]) => void;
  onBack: () => void;
  onRefresh: () => void;
  onRetry: () => void;
  onRerunDiarization: () => void;
  onRegenerate: () => void;
  onSaveTranscript: () => Promise<void> | void;
  onSubmitReview: () => Promise<void> | void;
  onSaveClinicalNote: (note: ClinicalNote) => Promise<void> | void;
}) {
  const disabled = data.status === "generating_note";
  const [editMode, setEditMode] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>("summary");
  const [isEditingNote, setIsEditingNote] = useState<boolean>(false);
  const [noteDraft, setNoteDraft] = useState<ClinicalNote | null>(null);
  const [savingNote, setSavingNote] = useState<boolean>(false);

  useEffect(() => {
    if (data.status === "awaiting_review") {
      setEditMode(true);
    } else {
      setEditMode(false);
    }
    setIsEditingNote(false);
    setNoteDraft(null);
  }, [data.session_id, data.status]);

  const handleStartEditNote = () => {
    if (data.clinical_note) {
      setNoteDraft(JSON.parse(JSON.stringify(data.clinical_note)));
      setIsEditingNote(true);
    }
  };

  const handleSaveNote = async () => {
    if (!noteDraft) return;
    setSavingNote(true);
    try {
      await onSaveClinicalNote(noteDraft);
      setIsEditingNote(false);
    } catch (e) {
      Message.error("保存修改失败: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setSavingNote(false);
    }
  };

  const handleCopy = () => {
    if (!data.clinical_note) return;
    let textToCopy = "";
    if (activeTab === "summary") {
      if (data.clinical_note.raw_text) {
        textToCopy = data.clinical_note.raw_text;
      } else {
        textToCopy = Object.entries(data.clinical_note.session_summary || {})
          .map(([key, val]) => `${key}:\n${val.map((v) => `- ${v}`).join("\n")}`)
          .join("\n\n");
      }
    } else {
      if (data.clinical_note.raw_text) {
        textToCopy = data.clinical_note.raw_text;
      } else {
        textToCopy = Object.entries(data.clinical_note.soap || {})
          .map(([key, val]) => `${key}:\n${val.map((v) => `- ${v}`).join("\n")}`)
          .join("\n\n");
      }
    }

    if (navigator.clipboard) {
      navigator.clipboard.writeText(textToCopy);
      Message.success("内容已复制到剪贴板");
    } else {
      Message.error("复制失败，您的浏览器不支持剪贴板操作");
    }
  };

  const setSegment = (index: number, patch: Partial<TranscriptSegment>) => {
    setTranscriptDraft(transcriptDraft.map((item, current) => (current === index ? { ...item, ...patch } : item)));
  };

  const [speakerModalVisible, setSpeakerModalVisible] = useState(false);
  const [speakerEditIndex, setSpeakerEditIndex] = useState<number | null>(null);
  const [speakerOldValue, setSpeakerOldValue] = useState("");
  const [speakerNewValue, setSpeakerNewValue] = useState("");
  const [batchUpdate, setBatchUpdate] = useState(false);

  const handleSpeakerChangeTrigger = (index: number, newSpeaker: string) => {
    const oldSpeaker = normalizeSpeaker(String(transcriptDraft[index]?.speaker || "未确认"));
    const normalizedNewSpeaker = normalizeSpeaker(newSpeaker);
    if (oldSpeaker === normalizedNewSpeaker) return;

    setSpeakerEditIndex(index);
    setSpeakerOldValue(oldSpeaker);
    setSpeakerNewValue(normalizedNewSpeaker);
    setBatchUpdate(false); // Default to unchecked as in mockup
    setSpeakerModalVisible(true);
  };

  const getSpeakerOptions = (currentSpeaker: string) => {
    const baseOptions = ["咨询师", "来访者", "未确认"];
    const activeSpeakers = Array.from(
      new Set(transcriptDraft.map((item) => normalizeSpeaker(String(item.speaker || "未确认"))))
    );
    const combined = Array.from(new Set([...baseOptions, ...activeSpeakers]));
    if (!combined.includes(currentSpeaker)) {
      combined.push(currentSpeaker);
    }
    return combined;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>, index: number, item: TranscriptSegment) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const textarea = e.currentTarget;
      const cursorPos = textarea.selectionStart;
      const text = item.text || "";
      const part1 = text.substring(0, cursorPos);
      const part2 = text.substring(cursorPos);

      const start = item.start;
      const end = item.end;
      const l1 = part1.length;
      const l2 = part2.length;
      let splitTime = start + (end - start) / 2;
      if (l1 + l2 > 0) {
        splitTime = start + (end - start) * (l1 / (l1 + l2));
      }
      splitTime = Math.round(splitTime * 100) / 100;

      const newSegments = [...transcriptDraft];
      newSegments[index] = {
        ...item,
        text: part1,
        end: splitTime,
      };
      const newSeg: TranscriptSegment = {
        start: splitTime,
        end: end,
        speaker: item.speaker,
        text: part2,
      };
      newSegments.splice(index + 1, 0, newSeg);
      setTranscriptDraft(newSegments);

      setTimeout(() => {
        const nextTextarea = document.querySelector(`.segment-textarea-${index + 1}`) as HTMLTextAreaElement | null;
        if (nextTextarea) {
          nextTextarea.focus();
          nextTextarea.setSelectionRange(0, 0);
        }
      }, 50);
    }
  };

  return (
    <section className="page session-page">
      <div className="detail-header">
        <Button
          icon={<IconArrowLeft />}
          shape="circle"
          onClick={onBack}
          aria-label="返回记录"
          title="返回记录"
        />
        <div className="detail-title">
          <Title heading={4}>{`${data.case_id || "-"}，${formatDateLabel(data.session_date)}`}</Title>
          <StatusTag status={data.status} label={data.progress_label} />
        </div>
        <Space wrap className="detail-actions">
          {data.status === "error" && <Button onClick={onRetry}>重新转写</Button>}
          {Boolean(data.transcript?.length && data.diarization_configured && !data.is_processing) && (
            <Button icon={<IconRobot />} onClick={onRerunDiarization}>
              重新自动整理
            </Button>
          )}
          <Button icon={<IconRefresh />} onClick={onRefresh}>
            刷新
          </Button>
        </Space>
      </div>

      <div className="detail-grid">
        <aside className="summary-pane">
          <Tabs
            activeTab={activeTab}
            onChange={(key) => setActiveTab(key)}
            extra={
              data.clinical_note && (
                <Space>
                  {isEditingNote ? (
                    <>
                      <Button size="small" type="primary" loading={savingNote} onClick={handleSaveNote}>
                        保存
                      </Button>
                      <Button size="small" type="text" onClick={() => setIsEditingNote(false)}>
                        取消
                      </Button>
                    </>
                  ) : (
                    <Button size="small" type="text" onClick={handleCopy}>
                      复制内容
                    </Button>
                  )}
                </Space>
              )
            }
          >
            <Tabs.TabPane key="summary" title="Summary">
              <ClinicalSummary
                note={isEditingNote ? noteDraft : (data.clinical_note || null)}
                status={data.status}
                isEditing={isEditingNote}
                onStartEdit={handleStartEditNote}
                onChange={setNoteDraft}
              />
            </Tabs.TabPane>
            <Tabs.TabPane key="soap" title="SOAP">
              <SoapNote
                note={isEditingNote ? noteDraft : (data.clinical_note || null)}
                status={data.status}
                isEditing={isEditingNote}
                onStartEdit={handleStartEditNote}
                onChange={setNoteDraft}
              />
            </Tabs.TabPane>
          </Tabs>
        </aside>

        <section className="transcript-pane">
          <div className="pane-head">
            <Title heading={5}>文字记录</Title>
          </div>
          {data.is_processing && !transcriptDraft.length ? (
            <div className="processing-empty-state">
              <Spin tip="正在生成文字记录，请稍候……" />
            </div>
          ) : !transcriptDraft.length ? (
            <Empty description="还没有逐字稿。" />
          ) : (
            <div className="transcript-list">
              {transcriptDraft.map((item, index) => (
                <div className="segment" key={`${item.start}-${item.end}-${index}`}>
                  <Text type="secondary" className="segment-time">
                    {formatTime(item.start)} - {formatTime(item.end)}
                  </Text>
                  {editMode ? (
                    <>
                      <Select
                        value={normalizeSpeaker(String(item.speaker || "Unknown"))}
                        disabled={disabled}
                        onChange={(speaker) => handleSpeakerChangeTrigger(index, speaker)}
                      >
                        {getSpeakerOptions(normalizeSpeaker(String(item.speaker || "未确认"))).map((opt) => (
                          <Select.Option key={opt} value={opt}>
                            {opt}
                          </Select.Option>
                        ))}
                      </Select>
                      <TextArea
                        value={item.text || ""}
                        className={`segment-textarea-${index}`}
                        disabled={disabled}
                        autoSize={{ minRows: 2, maxRows: 8 }}
                        onChange={(text) => setSegment(index, { text })}
                        onKeyDown={(e) => handleKeyDown(e, index, item)}
                      />
                    </>
                  ) : (
                    <>
                      <div className="segment-speaker-readonly">
                        <Tag color={item.speaker === "咨询师" ? "arcoblue" : item.speaker === "来访者" ? "green" : "gray"}>
                          {item.speaker || "未确认"}
                        </Tag>
                      </div>
                      <div className="segment-text-readonly">
                        <Text>{item.text}</Text>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          )}
          <div className="review-bar">
            <div className="review-bar-left" />
            <Space>
              {!editMode ? (
                (data.status === "awaiting_review" || data.status === "complete" || data.status === "error") && (
                  <Button type="primary" onClick={() => setEditMode(true)}>
                    编辑
                  </Button>
                )
              ) : (
                data.status === "awaiting_review" ? (
                  <Button type="primary" onClick={onSubmitReview}>
                    生成记录
                  </Button>
                ) : (
                  <>
                    <Button onClick={async () => {
                      await onSaveTranscript();
                      setEditMode(false);
                    }}>
                      保存
                    </Button>
                    <Button type="primary" onClick={onSubmitReview}>
                      保存并重新生成记录
                    </Button>
                  </>
                )
              )}
            </Space>
          </div>
        </section>
      </div>
      <Modal
        title="修改说话人"
        visible={speakerModalVisible}
        onCancel={() => {
          setSpeakerModalVisible(false);
          setSpeakerEditIndex(null);
        }}
        onOk={() => {
          if (speakerEditIndex === null) return;
          if (batchUpdate) {
            setTranscriptDraft(
              transcriptDraft.map((item) =>
                normalizeSpeaker(String(item.speaker || "未确认")) === speakerOldValue
                  ? { ...item, speaker: speakerNewValue }
                  : item
              )
            );
          } else {
            setTranscriptDraft(
              transcriptDraft.map((item, current) =>
                current === speakerEditIndex ? { ...item, speaker: speakerNewValue } : item
              )
            );
          }
          setSpeakerModalVisible(false);
          setSpeakerEditIndex(null);
        }}
        okText="完成"
        cancelText="取消"
        unmountOnExit
      >
        <div style={{ marginBottom: 20 }}>
          <Text>将该片段的说话人修改为：</Text>
          <Tag color="arcoblue" style={{ marginLeft: 8, fontSize: 14, padding: "4px 8px" }}>{speakerNewValue}</Tag>
        </div>
        <div style={{ padding: "8px 0" }}>
          <Checkbox checked={batchUpdate} onChange={(checked) => setBatchUpdate(checked)}>
            {`批量修改全文中所有 ${
              transcriptDraft.filter(
                (item) => normalizeSpeaker(String(item.speaker || "未确认")) === speakerOldValue
              ).length
            } 处 “${speakerOldValue}”`}
          </Checkbox>
        </div>
      </Modal>
    </section>
  );
}

function ClinicalSummary({
  note,
  status,
  isEditing,
  onStartEdit,
  onChange,
}: {
  note: ClinicalNote | null;
  status?: string;
  isEditing?: boolean;
  onStartEdit?: () => void;
  onChange?: (note: ClinicalNote) => void;
}) {
  if (status === "generating_note" || status === "queued_note_generation") {
    return (
      <div className="processing-empty-state">
        <Spin tip="正在生成 Session Summary，请稍候……" />
      </div>
    );
  }
  if (!note) return <Empty description="确认角色后，系统会生成 Session Summary。" />;

  const summary = note.session_summary || {};
  const keys = Object.keys(summary);

  if (isEditing) {
    if (note.raw_text !== undefined && note.raw_text !== null) {
      return (
        <div style={{ padding: "8px 16px" }}>
          <Title heading={6} style={{ marginBottom: "8px" }}>原始文本</Title>
          <TextArea
            autoSize={{ minRows: 10, maxRows: 25 }}
            value={note.raw_text}
            onChange={(val) => {
              if (onChange) onChange({ ...note, raw_text: val });
            }}
          />
        </div>
      );
    }
    if (keys.length === 0) {
      return (
        <div style={{ padding: "8px 16px" }}>
          <Button
            type="outline"
            size="small"
            onClick={() => {
              if (onChange) {
                onChange({
                  ...note,
                  session_summary: { "本次会谈整体摘要": [""] }
                });
              }
            }}
          >
            添加摘要字段
          </Button>
        </div>
      );
    }
    return (
      <div className="note-stack">
        {keys.map((key) => {
          const val = (summary[key] || []).join("\n");
          return (
            <section className="note-section" key={key}>
              <Title heading={6}>{key}</Title>
              <TextArea
                autoSize={{ minRows: 4, maxRows: 15 }}
                value={val}
                onChange={(newVal) => {
                  const updatedSummary = { ...summary };
                  updatedSummary[key] = newVal.split("\n");
                  if (onChange) {
                    onChange({ ...note, session_summary: updatedSummary });
                  }
                }}
              />
            </section>
          );
        })}
      </div>
    );
  }

  if (note.raw_text) {
    return (
      <div
        onClick={onStartEdit}
        style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, padding: "8px 16px", cursor: "pointer" }}
        title="点击内容进行编辑"
      >
        <Text>{note.raw_text}</Text>
      </div>
    );
  }

  if (keys.length === 0) {
    return <Empty description="未输出 Session Summary 记录内容。" />;
  }

  return (
    <div
      className="note-stack clickable-note-stack"
      onClick={onStartEdit}
      style={{ cursor: "pointer" }}
      title="点击内容进行编辑"
    >
      {keys.map((key) => (
        <section className="note-section" key={key}>
          <Title heading={6}>{key}</Title>
          <NoteList items={summary[key]} />
        </section>
      ))}
    </div>
  );
}

function SoapNote({
  note,
  status,
  isEditing,
  onStartEdit,
  onChange,
}: {
  note: ClinicalNote | null;
  status?: string;
  isEditing?: boolean;
  onStartEdit?: () => void;
  onChange?: (note: ClinicalNote) => void;
}) {
  if (status === "generating_note" || status === "queued_note_generation") {
    return (
      <div className="processing-empty-state">
        <Spin tip="正在生成 SOAP 记录，请稍候……" />
      </div>
    );
  }
  if (!note) return <Empty description="确认角色后，系统会生成 SOAP 记录。" />;

  const soap = note.soap || {};
  const keys = ["S (主观感觉)", "O (客观表现)", "A (评估分析)", "P (后续计划)"];

  if (isEditing) {
    if (note.raw_text !== undefined && note.raw_text !== null) {
      return (
        <div style={{ padding: "8px 16px" }}>
          <Title heading={6} style={{ marginBottom: "8px" }}>原始文本</Title>
          <TextArea
            autoSize={{ minRows: 10, maxRows: 25 }}
            value={note.raw_text}
            onChange={(val) => {
              if (onChange) onChange({ ...note, raw_text: val });
            }}
          />
        </div>
      );
    }

    // Ensure all keys exist in draft
    const ensureSoap = { ...soap };
    keys.forEach((k) => {
      if (!ensureSoap[k]) ensureSoap[k] = [];
    });

    return (
      <div className="note-stack">
        {keys.map((key) => {
          const val = (ensureSoap[key] || []).join("\n");
          return (
            <section className="note-section" key={key}>
              <Title heading={6}>{key}</Title>
              <TextArea
                autoSize={{ minRows: 2, maxRows: 8 }}
                value={val}
                onChange={(newVal) => {
                  const updatedSoap = { ...ensureSoap };
                  updatedSoap[key] = newVal.split("\n").filter((line) => line.trim() !== "");
                  if (onChange) {
                    onChange({ ...note, soap: updatedSoap });
                  }
                }}
              />
            </section>
          );
        })}
      </div>
    );
  }

  if (note.raw_text) {
    return (
      <div
        onClick={onStartEdit}
        style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, padding: "8px 16px", cursor: "pointer" }}
        title="点击内容进行编辑"
      >
        <Text>{note.raw_text}</Text>
      </div>
    );
  }

  const activeSoapKeys = Object.keys(soap);
  if (activeSoapKeys.length === 0) {
    return <Empty description="未输出 SOAP 记录内容。" />;
  }

  return (
    <div
      className="note-stack clickable-note-stack"
      onClick={onStartEdit}
      style={{ cursor: "pointer" }}
      title="点击内容进行编辑"
    >
      {activeSoapKeys.map((key) => (
        <section className="note-section" key={key}>
          <Title heading={6}>{key}</Title>
          <NoteList items={soap[key]} />
        </section>
      ))}
    </div>
  );
}
