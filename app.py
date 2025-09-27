import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Label, Button, Canvas, Scrollbar, Toplevel, Listbox, MULTIPLE
from tkinter import ttk
from PIL import Image, ImageTk
import os
import shutil
import face_recognition
import numpy as np
import threading

class FaceSelectionDialog:
    def __init__(self, parent, face_images, face_locations, full_image_path):
        self.parent = parent
        self.face_images = face_images
        self.face_locations = face_locations
        self.full_image_path = full_image_path
        self.selected_faces = set()  # Track multiple selected faces
        
        self.dialog = Toplevel(parent)
        self.dialog.title("Select Faces to Use as Reference")
        self.dialog.geometry("800x600")
        self.dialog.configure(bg="#f0f0f0")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title and instructions
        title = Label(self.dialog, text="Select Faces to Use as New Reference", 
                     font=("Arial", 14, "bold"), bg="#f0f0f0")
        title.pack(pady=10)
        
        instruction = Label(self.dialog, 
                           text="Click on faces to select/deselect. All selected faces will be used for similarity matching.", 
                           font=("Arial", 10), bg="#f0f0f0", wraplength=700)
        instruction.pack(pady=5)
        
        # Full image preview
        full_img_frame = Frame(self.dialog, bg="#ffffff", relief="sunken", borderwidth=1)
        full_img_frame.pack(pady=10, padx=20, fill="x")
        
        try:
            full_img = Image.open(self.full_image_path)
            full_img.thumbnail((300, 200), Image.Resampling.LANCZOS)
            self.full_photo = ImageTk.PhotoImage(full_img)
            full_img_label = Label(full_img_frame, image=self.full_photo, bg="#ffffff")
            full_img_label.pack(pady=10)
        except Exception as e:
            Label(full_img_frame, text="Could not load full image", bg="#ffffff").pack(pady=10)
        
        # Faces selection area
        faces_frame = Frame(self.dialog, bg="#f0f0f0")
        faces_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Canvas for scrollable face previews
        canvas = Canvas(faces_frame, bg="#f0f0f0")
        scrollbar = Scrollbar(faces_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#f0f0f0")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Display face previews
        self.face_photos = []
        self.face_buttons = []
        
        cols = 4
        for idx, face_img in enumerate(self.face_images):
            face_frame = Frame(scrollable_frame, bg="#ffffff", relief="raised", 
                             borderwidth=1, padx=5, pady=5)
            face_frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10, sticky="nsew")
            
            # Create thumbnail
            face_img.thumbnail((120, 120), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(face_img)
            self.face_photos.append(photo)
            
            # Clickable face button
            face_btn = Button(face_frame, image=photo, 
                            command=lambda i=idx: self.toggle_face_selection(i),
                            relief="flat", bg="#ffffff", cursor="hand2")
            face_btn.pack(pady=5)
            self.face_buttons.append(face_btn)
            
            # Face number label
            face_label = Label(face_frame, text=f"Face {idx + 1}", 
                             font=("Arial", 9), bg="#ffffff")
            face_label.pack(pady=2)
        
        # Button frame
        button_frame = Frame(self.dialog, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        # Select All button
        select_all_btn = Button(button_frame, text="Select All Faces", 
                              command=self.select_all_faces,
                              font=("Arial", 10), bg="#2196F3", fg="white", padx=15)
        select_all_btn.pack(side="left", padx=5)
        
        # Clear Selection button
        clear_btn = Button(button_frame, text="Clear Selection", 
                         command=self.clear_selection,
                         font=("Arial", 10), bg="#FF9800", fg="white", padx=15)
        clear_btn.pack(side="left", padx=5)
        
        # OK button (initially disabled)
        self.ok_btn = Button(button_frame, text="Use Selected Faces", 
                           command=self.ok_clicked,
                           font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
                           state="disabled", padx=20, pady=5)
        self.ok_btn.pack(side="left", padx=10)
        
        # Cancel button
        cancel_btn = Button(button_frame, text="Cancel", 
                          command=self.cancel_clicked,
                          font=("Arial", 10), bg="#f44336", fg="white", padx=15)
        cancel_btn.pack(side="left", padx=5)
        
    def toggle_face_selection(self, index):
        if index in self.selected_faces:
            self.selected_faces.remove(index)
            self.face_buttons[index].config(relief="flat", borderwidth=0, bg="#ffffff")
        else:
            self.selected_faces.add(index)
            self.face_buttons[index].config(relief="solid", borderwidth=3, bg="#e3f2fd")
        
        # Update OK button state
        if self.selected_faces:
            self.ok_btn.config(state="normal", bg="#45a049")
        else:
            self.ok_btn.config(state="disabled", bg="#4CAF50")
    
    def select_all_faces(self):
        self.selected_faces = set(range(len(self.face_images)))
        for i, btn in enumerate(self.face_buttons):
            btn.config(relief="solid", borderwidth=3, bg="#e3f2fd")
        self.ok_btn.config(state="normal", bg="#45a049")
    
    def clear_selection(self):
        self.selected_faces.clear()
        for btn in self.face_buttons:
            btn.config(relief="flat", borderwidth=0, bg="#ffffff")
        self.ok_btn.config(state="disabled", bg="#4CAF50")
    
    def ok_clicked(self):
        if self.selected_faces:
            self.dialog.destroy()
    
    def cancel_clicked(self):
        self.selected_faces.clear()
        self.dialog.destroy()
    
    def wait_for_selection(self):
        self.parent.wait_window(self.dialog)
        return sorted(self.selected_faces) if self.selected_faces else None

class FileManagementDialog:
    def __init__(self, parent, selected_images, folder_path):
        self.parent = parent
        self.selected_images = selected_images
        self.folder_path = folder_path
        self.operation = None
        
        self.dialog = Toplevel(parent)
        self.dialog.title("File Management")
        self.dialog.geometry("600x500")
        self.dialog.configure(bg="#f0f0f0")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = Label(self.dialog, text="File Management", 
                     font=("Arial", 16, "bold"), bg="#f0f0f0")
        title.pack(pady=10)
        
        # Selected files info
        info_text = f"Selected {len(self.selected_images)} files:"
        info_label = Label(self.dialog, text=info_text, 
                          font=("Arial", 11), bg="#f0f0f0")
        info_label.pack(pady=5)
        
        # File listbox
        listbox_frame = Frame(self.dialog, bg="#f0f0f0")
        listbox_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.listbox = Listbox(listbox_frame, selectmode=MULTIPLE, 
                              font=("Arial", 10), bg="white")
        scrollbar = Scrollbar(listbox_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        for img_path in self.selected_images:
            self.listbox.insert(tk.END, os.path.basename(img_path))
        
        # Select all files by default
        for i in range(len(self.selected_images)):
            self.listbox.select_set(i)
        
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Operation selection
        op_frame = Frame(self.dialog, bg="#f0f0f0")
        op_frame.pack(pady=10)
        
        Label(op_frame, text="Operation:", font=("Arial", 11), 
             bg="#f0f0f0").pack(side="left", padx=5)
        
        self.op_var = tk.StringVar(value="copy")
        copy_btn = tk.Radiobutton(op_frame, text="Copy", variable=self.op_var, 
                                value="copy", bg="#f0f0f0", font=("Arial", 10))
        copy_btn.pack(side="left", padx=10)
        
        cut_btn = tk.Radiobutton(op_frame, text="Cut", variable=self.op_var, 
                               value="cut", bg="#f0f0f0", font=("Arial", 10))
        cut_btn.pack(side="left", padx=10)
        
        # Destination selection
        dest_frame = Frame(self.dialog, bg="#f0f0f0")
        dest_frame.pack(pady=10, padx=20, fill="x")
        
        Label(dest_frame, text="Destination Folder:", font=("Arial", 11), 
             bg="#f0f0f0").pack(anchor="w")
        
        dest_btn_frame = Frame(dest_frame, bg="#f0f0f0")
        dest_btn_frame.pack(fill="x", pady=5)
        
        self.dest_var = tk.StringVar()
        dest_entry = tk.Entry(dest_btn_frame, textvariable=self.dest_var, 
                             font=("Arial", 10), width=40)
        dest_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        browse_btn = Button(dest_btn_frame, text="Browse", 
                          command=self.browse_destination,
                          font=("Arial", 9), bg="#2196F3", fg="white")
        browse_btn.pack(side="right")
        
        # Button frame
        button_frame = Frame(self.dialog, bg="#f0f0f0")
        button_frame.pack(pady=15)
        
        # Execute button
        execute_btn = Button(button_frame, text="Execute Operation", 
                           command=self.execute_operation,
                           font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
                           padx=20, pady=5)
        execute_btn.pack(side="left", padx=10)
        
        # Cancel button
        cancel_btn = Button(button_frame, text="Cancel", 
                          command=self.cancel_operation,
                          font=("Arial", 10), bg="#f44336", fg="white", padx=15)
        cancel_btn.pack(side="left", padx=10)
        
    def browse_destination(self):
        folder = filedialog.askdirectory(title="Select destination folder")
        if folder:
            self.dest_var.set(folder)
    
    def execute_operation(self):
        if not self.dest_var.get():
            messagebox.showwarning("Warning", "Please select a destination folder.")
            return
        
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select files to operate on.")
            return
        
        operation = self.op_var.get()
        dest_folder = self.dest_var.get()
        
        try:
            success_count = 0
            for idx in selected_indices:
                src_path = self.selected_images[idx]
                filename = os.path.basename(src_path)
                dest_path = os.path.join(dest_folder, filename)
                
                if operation == "copy":
                    shutil.copy2(src_path, dest_path)
                    success_count += 1
                else:  # cut
                    shutil.move(src_path, dest_path)
                    success_count += 1
            
            messagebox.showinfo("Success", 
                              f"Successfully {operation}ed {success_count} files to:\n{dest_folder}")
            self.operation = operation
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during file operation: {e}")
    
    def cancel_operation(self):
        self.operation = None
        self.dialog.destroy()
    
    def wait_for_operation(self):
        self.parent.wait_window(self.dialog)
        return self.operation

class FaceSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Similarity Finder with File Management")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.folder_path = ""
        self.folder_face_data = []
        self.tolerance = tk.DoubleVar(value=0.5)
        self.reference_encodings = []
        self.current_matches = []
        self.search_history = []
        self.selected_images = set()

        # Header
        header = Label(root, text="Face Similarity Finder with File Management", 
                      font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333")
        header.pack(pady=20)

        # Controls Frame
        controls = Frame(root, bg="#f0f0f0")
        controls.pack(pady=10)

        # Row 1: Main controls
        self.btn_select_folder = Button(controls, text="Select Folder with Images", 
                                       command=self.select_folder, font=("Arial", 12, "bold"),
                                       bg="#4CAF50", fg="white", padx=20, pady=10)
        self.btn_select_folder.grid(row=0, column=0, padx=20, pady=5)

        self.btn_select_query = Button(controls, text="Select Reference Image", 
                                      command=self.select_query_image, font=("Arial", 12, "bold"),
                                      bg="#2196F3", fg="white", padx=20, pady=10)
        self.btn_select_query.grid(row=0, column=1, padx=20, pady=5)
        self.btn_select_query.config(state="disabled")

        # Tolerance Slider
        Label(controls, text="Tolerance (0.4-0.6):", font=("Arial", 12), 
             bg="#f0f0f0").grid(row=0, column=2, padx=10, pady=5)
        tolerance_slider = tk.Scale(controls, from_=0.4, to=0.6, variable=self.tolerance, 
                                   orient="horizontal", length=200, resolution=0.01,
                                   bg="#f0f0f0", font=("Arial", 10))
        tolerance_slider.grid(row=0, column=3, padx=10, pady=5)

        # Row 2: Navigation and file management
        self.btn_back = Button(controls, text="← Back to Previous Search", 
                              command=self.navigate_back, font=("Arial", 10),
                              bg="#607D8B", fg="white", padx=15, pady=5, state="disabled")
        self.btn_back.grid(row=1, column=0, padx=20, pady=5)

        self.btn_file_manage = Button(controls, text="📁 Manage Selected Files", 
                                     command=self.manage_files, font=("Arial", 10, "bold"),
                                     bg="#FF9800", fg="white", padx=15, pady=5, state="disabled")
        self.btn_file_manage.grid(row=1, column=1, padx=20, pady=5)

        self.btn_select_all = Button(controls, text="Select All Results", 
                                    command=self.select_all_results, font=("Arial", 10),
                                    bg="#9C27B0", fg="white", padx=15, pady=5, state="disabled")
        self.btn_select_all.grid(row=1, column=2, padx=20, pady=5)

        self.btn_clear_selection = Button(controls, text="Clear Selection", 
                                         command=self.clear_selection, font=("Arial", 10),
                                         bg="#f44336", fg="white", padx=15, pady=5, state="disabled")
        self.btn_clear_selection.grid(row=1, column=3, padx=20, pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.stop()

        # Status Label
        self.status_label = Label(root, text="", font=("Arial", 11), bg="#f0f0f0", fg="#666")
        self.status_label.pack(pady=5)

        # Matches Frame
        self.matches_frame = Frame(root, bg="#ffffff", relief="sunken", borderwidth=2)
        self.matches_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.match_photos = []
        self.image_frames = {}

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing photos")
        if folder:
            self.folder_path = folder
            self.reference_encodings.clear()
            self.search_history.clear()
            self.selected_images.clear()
            self.update_file_management_buttons()
            self.btn_back.config(state="disabled")
            self.status_label.config(text="Loading and processing images...")
            self.progress.start()
            threading.Thread(target=self.load_folder_faces, daemon=True).start()

    def load_folder_faces(self):
        self.folder_face_data.clear()

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_ext)]
        
        processed = 0
        total = len(files)

        for file in files:
            path = os.path.join(self.folder_path, file)
            try:
                img = face_recognition.load_image_file(path)
                face_locations = face_recognition.face_locations(img)
                
                face_encodings = []
                for location in face_locations:
                    top, right, bottom, left = location
                    face_img = img[top:bottom, left:right]
                    encoding = face_recognition.face_encodings(face_img)
                    if encoding:
                        face_encodings.append(encoding[0])
                    else:
                        encoding = face_recognition.face_encodings(img, [location])
                        if encoding:
                            face_encodings.append(encoding[0])
                
                if face_encodings:
                    self.folder_face_data.append({
                        'path': path,
                        'encodings': face_encodings,
                        'locations': face_locations
                    })
                
                processed += 1
                if processed % 5 == 0:
                    self.root.after(0, lambda p=processed, t=total: 
                                  self.status_label.config(text=f"Processed {p}/{t} images..."))
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")

        self.root.after(0, self.finish_loading)

    def finish_loading(self):
        self.progress.stop()
        total_faces = sum(len(data['encodings']) for data in self.folder_face_data)
        messagebox.showinfo("Folder Loaded", 
                          f"Processed {len(self.folder_face_data)} images with {total_faces} total faces detected.")
        self.btn_select_query.config(state="normal")
        self.status_label.config(text=f"Ready - {len(self.folder_face_data)} images loaded with {total_faces} faces")

    def select_query_image(self):
        path = filedialog.askopenfilename(title="Select reference image",
                                        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return

        try:
            img = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(img)
            
            face_encodings = []
            for location in face_locations:
                top, right, bottom, left = location
                face_img = img[top:bottom, left:right]
                encoding = face_recognition.face_encodings(face_img)
                if encoding:
                    face_encodings.append(encoding[0])
                else:
                    encoding = face_recognition.face_encodings(img, [location])
                    if encoding:
                        face_encodings.append(encoding[0])
            
            if not face_encodings:
                messagebox.showwarning("No face detected", "No faces detected in reference image.")
                return
            
            if self.reference_encodings:
                self.search_history.append(self.reference_encodings.copy())
                self.btn_back.config(state="normal")
            
            if len(face_encodings) == 1:
                self.reference_encodings = face_encodings
                self.start_matching_process()
            else:
                self.show_face_selection_dialog(path, face_locations, face_encodings)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error processing reference image: {e}")

    def show_face_selection_dialog(self, image_path, face_locations, face_encodings):
        try:
            pil_image = Image.open(image_path)
            face_images = []
            for location in face_locations:
                top, right, bottom, left = location
                margin = 20
                expanded_top = max(0, top - margin)
                expanded_bottom = min(pil_image.height, bottom + margin)
                expanded_left = max(0, left - margin)
                expanded_right = min(pil_image.width, right + margin)
                face_img = pil_image.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))
                face_images.append(face_img)
            
            dialog = FaceSelectionDialog(self.root, face_images, face_locations, image_path)
            selected_indices = dialog.wait_for_selection()
            
            if selected_indices is not None:
                if self.reference_encodings:
                    self.search_history.append(self.reference_encodings.copy())
                    self.btn_back.config(state="normal")
                
                selected_encodings = [face_encodings[i] for i in selected_indices]
                self.reference_encodings = selected_encodings
                self.start_matching_process()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error showing face selection: {e}")

    def start_matching_process(self):
        self.selected_images.clear()
        self.update_file_management_buttons()
        self.status_label.config(text="Finding similar faces...")
        self.progress.start()
        threading.Thread(target=self.process_matches, daemon=True).start()

    def process_matches(self):
        matches = self.find_similar_faces(self.reference_encodings, self.tolerance.get())
        self.current_matches = matches
        self.root.after(0, lambda: self.show_matches(matches))

    def find_similar_faces(self, query_encodings, tolerance):
        matched_paths = []
        
        for image_data in self.folder_face_data:
            path = image_data['path']
            face_encodings = image_data['encodings']
            
            best_distance = float('inf')
            best_face_index = -1
            
            for i, face_encoding in enumerate(face_encodings):
                for ref_encoding in query_encodings:
                    distance = np.linalg.norm(face_encoding - ref_encoding)
                    if distance < best_distance:
                        best_distance = distance
                        best_face_index = i
            
            if best_distance <= tolerance and best_face_index != -1:
                matched_paths.append({
                    'path': path,
                    'distance': best_distance,
                    'face_index': best_face_index,
                    'total_faces_in_image': len(face_encodings),
                    'total_ref_faces': len(query_encodings)
                })
        
        matched_paths.sort(key=lambda x: x['distance'])
        return matched_paths

    def show_matches(self, matched_paths):
        self.progress.stop()
        
        for widget in self.matches_frame.winfo_children():
            widget.destroy()
        self.match_photos.clear()
        self.image_frames.clear()

        if not matched_paths:
            Label(self.matches_frame, text="No matching faces found.", 
                 font=("Arial", 14), bg="#ffffff").pack(pady=20)
            self.status_label.config(text="No matches found")
            self.btn_select_all.config(state="disabled")
            return

        ref_faces_info = f" ({len(self.reference_encodings)} reference faces)" if len(self.reference_encodings) > 1 else ""
        self.status_label.config(text=f"Found {len(matched_paths)} matching images{ref_faces_info} - Click images to select or use as reference")
        self.btn_select_all.config(state="normal")

        header_text = f"Found {len(matched_paths)} matching images{ref_faces_info}:"
        header = Label(self.matches_frame, text=header_text, 
                      font=("Arial", 16, "bold"), bg="#ffffff")
        header.pack(pady=10)

        instruction = Label(self.matches_frame, 
                          text="💡 Click image to select for file management | Double-click to use as new reference", 
                          font=("Arial", 10), bg="#ffffff", fg="#2196F3")
        instruction.pack(pady=5)

        canvas = Canvas(self.matches_frame, bg="#ffffff")
        scrollbar = Scrollbar(self.matches_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#ffffff")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        cols = 4
        for idx, match_data in enumerate(matched_paths):
            path = match_data['path']
            distance = match_data['distance']
            face_index = match_data['face_index']
            total_faces = match_data['total_faces_in_image']

            img_frame = Frame(scrollable_frame, bg="#f0f0f0", relief="raised", 
                             borderwidth=2, padx=5, pady=5)
            img_frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10, sticky="nsew")
            self.image_frames[path] = img_frame

            try:
                img = Image.open(path)
                img_thumbnail = img.copy()
                img_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img_thumbnail)
                self.match_photos.append(photo)
                
                img_label = Label(img_frame, image=photo, bg="#f0f0f0", cursor="hand2")
                img_label.pack(pady=5)
                
                # Single click for selection, double-click for reference
                img_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                img_label.bind("<Double-Button-1>", lambda e, p=path: self.use_image_as_new_reference(p))
                img_frame.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                img_frame.bind("<Double-Button-1>", lambda e, p=path: self.use_image_as_new_reference(p))

                similarity = (1 - distance) * 100
                info_text = f"{os.path.basename(path)}\nSimilarity: {similarity:.1f}%"
                if total_faces > 1:
                    info_text += f"\nFaces: {total_faces}"
                
                file_label = Label(img_frame, text=info_text, 
                                 font=("Arial", 10), bg="#f0f0f0", wraplength=200, 
                                 justify="center", cursor="hand2")
                file_label.pack(pady=5)
                file_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                file_label.bind("<Double-Button-1>", lambda e, p=path: self.use_image_as_new_reference(p))
                
            except Exception as e:
                error_label = Label(img_frame, text=f"Error loading image\n{os.path.basename(path)}", 
                                  font=("Arial", 10), bg="#f0f0f0", fg="red")
                error_label.pack(pady=20)

        self.update_file_management_buttons()

    def toggle_image_selection(self, image_path):
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
            self.image_frames[image_path].config(bg="#f0f0f0")
        else:
            self.selected_images.add(image_path)
            self.image_frames[image_path].config(bg="#e3f2fd")
        
        self.update_file_management_buttons()

    def use_image_as_new_reference(self, image_path):
        image_data = next((data for data in self.folder_face_data if data['path'] == image_path), None)
        if not image_data or not image_data['encodings']:
            messagebox.showerror("Error", "Could not find face data for selected image.")
            return
        
        if len(image_data['encodings']) == 1:
            # Single face - use directly
            if self.reference_encodings:
                self.search_history.append(self.reference_encodings.copy())
                self.btn_back.config(state="normal")
            self.reference_encodings = image_data['encodings']
            self.start_matching_process()
        else:
            # Multiple faces - show selection dialog
            self.show_face_selection_dialog(image_path, image_data['locations'], image_data['encodings'])

    def navigate_back(self):
        if self.search_history:
            previous_encodings = self.search_history.pop()
            self.reference_encodings = previous_encodings
            if not self.search_history:
                self.btn_back.config(state="disabled")
            self.status_label.config(text="Returning to previous search...")
            self.start_matching_process()

    def select_all_results(self):
        if self.current_matches:
            self.selected_images.clear()
            for match in self.current_matches:
                self.selected_images.add(match['path'])
            for path in self.selected_images:
                if path in self.image_frames:
                    self.image_frames[path].config(bg="#e3f2fd")
            self.update_file_management_buttons()

    def clear_selection(self):
        self.selected_images.clear()
        for path in self.image_frames:
            self.image_frames[path].config(bg="#f0f0f0")
        self.update_file_management_buttons()

    def update_file_management_buttons(self):
        if self.selected_images:
            self.btn_file_manage.config(state="normal")
            self.btn_clear_selection.config(state="normal")
            self.status_label.config(text=f"Selected {len(self.selected_images)} images - Ready for file management")
        else:
            self.btn_file_manage.config(state="disabled")
            self.btn_clear_selection.config(state="disabled")

    def manage_files(self):
        if not self.selected_images:
            messagebox.showwarning("Warning", "No images selected.")
            return
        
        dialog = FileManagementDialog(self.root, list(self.selected_images), self.folder_path)
        operation = dialog.wait_for_operation()
        
        if operation == "cut":
            # Remove cut files from current matches and selection
            self.selected_images.clear()
            self.update_file_management_buttons()
            # Reload folder to reflect changes
            self.status_label.config(text="Reloading folder after file operation...")
            self.progress.start()
            threading.Thread(target=self.load_folder_faces, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSimilarityApp(root)
    root.mainloop()