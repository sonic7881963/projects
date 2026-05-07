
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level5/global - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'ac1f3CNYoBK/rt2tvbz7Ys1', 'global - 001');
// 3.16小游戏/command_TypeScript/level5/global - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var gloabl = /** @class */ (function (_super) {
    __extends(gloabl, _super);
    function gloabl() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    gloabl_1 = gloabl;
    gloabl.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    gloabl.prototype.onEventStart = function () {
        cc.log("click");
        gloabl_1.attack = true;
    };
    var gloabl_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    gloabl.attack = false;
    gloabl.minion_attack = false;
    gloabl.main_hp = 163;
    gloabl.minion1_hp = 280;
    gloabl.minion2_hp = 163;
    gloabl.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], gloabl.prototype, "label", void 0);
    __decorate([
        property
    ], gloabl.prototype, "text", void 0);
    gloabl = gloabl_1 = __decorate([
        ccclass
    ], gloabl);
    return gloabl;
}(cc.Component));
exports.default = gloabl;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDVcXGdsb2JhbCAtIDAwMS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQTRCQztRQXpCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBc0IzQixDQUFDO2VBNUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxFQUFFLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ2hCLFFBQU0sQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO0lBQ3pCLENBQUM7O0lBbEJELHdCQUF3QjtJQUV4QixlQUFlO0lBRUYsYUFBTSxHQUFZLEtBQUssQ0FBQztJQUN4QixvQkFBYSxHQUFZLEtBQUssQ0FBQztJQUMvQixjQUFPLEdBQVcsR0FBRyxDQUFDO0lBQ3RCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBZHRDO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7eUNBQ0k7SUFHdkI7UUFEQyxRQUFRO3dDQUNjO0lBTk4sTUFBTTtRQUQxQixPQUFPO09BQ2EsTUFBTSxDQTRCMUI7SUFBRCxhQUFDO0NBNUJELEFBNEJDLENBNUJtQyxFQUFFLENBQUMsU0FBUyxHQTRCL0M7a0JBNUJvQixNQUFNIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBnbG9hYmwgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSAyODA7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMl9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjNfaHA6IG51bWJlciA9IDE2MztcclxuICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMubm9kZS5vbigndG91Y2hzdGFydCcsIHRoaXMub25FdmVudFN0YXJ0LCB0aGlzKTtcclxuICAgIH1cclxuICAgIG9uRXZlbnRTdGFydCgpIHtcclxuICAgICAgICBjYy5sb2coXCJjbGlja1wiKTtcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG59XHJcbiJdfQ==